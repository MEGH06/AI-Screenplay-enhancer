import json
import re
from typing import Dict, List
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from file_handler import extract_pdf_text, extract_txt_text, extract_docx_text

load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")  

# Data structures from ai_screenplay_megh.py
@dataclass
class Character:
    name: str
    speaking_style: str = ""
    emotional_arc: str = ""

@dataclass
class Scene:
    characters: List[str]
    dialogues: List[Dict[str, str]]
    actions: List[str]
    tone_tags: List[str] = None
    scene_id: int = 0
    
    def __post_init__(self):
        if self.tone_tags is None:
            self.tone_tags = []

@dataclass
class State:
    raw_script: str = ""
    scenes: List[Scene] = None
    characters: Dict[str, Character] = None
    current_round: int = 0
    max_rounds: int = 3
    issues_found: List[str] = None
    final_script: str = ""
    scene_chunks: List[Dict] = None

  
    
    def __post_init__(self):
        if self.scenes is None:
            self.scenes = []
        if self.characters is None:
            self.characters = {}
        if self.issues_found is None:
            self.issues_found = []
        if self.scene_chunks is None:
            self.scene_chunks = []

# Initialize Groq
llm= ChatGroq(temperature=0.7, model_name="llama3-70b-8192")

def simple_scene_splitter(script_text: str) -> List[Dict]:
    scene_heading_pattern = r"(?:^|\n)((?:INT\.|EXT\.|SCENE\s+\d+)[^\n]*)\n"
    splits = list(re.finditer(scene_heading_pattern, script_text, re.IGNORECASE))
    scenes = []

    for i, match in enumerate(splits):
        start = match.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(script_text)
        heading = match.group(1).strip()
        scene_text = script_text[start:end].strip()
        
        # Modified regex to better catch character names
        characters = list(set(
            re.findall(r"^([A-Z][A-Z0-9_ ]+)(?:\s*:|\s+(?:sits|stands|walks|runs|enters|exits))", 
                      scene_text, 
                      re.MULTILINE)
        ))

        scenes.append({
            "scene_id": i + 1,
            "scene_text": scene_text,
            "characters": characters
        })
    return scenes

# Chunk maker logic from ai_screenplay_new.py
def chunk_scenes_with_text(script_text: str, threshold: int = 10) -> List[Dict]:
    scenes = simple_scene_splitter(script_text)
    chunks = []
    total_scenes = len(scenes)

    if total_scenes <= threshold:
        for i, scene in enumerate(scenes):
            chunks.append({
                "chunk_id": i + 1,
                "chunk_scenes": [scene["scene_id"]],
                "characters": scene["characters"],
                "scene_texts": [scene["scene_text"]]
            })
    else:
        chunk_size = total_scenes // threshold
        remainder = total_scenes % threshold
        start = 0
        for i in range(threshold):
            end = start + chunk_size + (1 if i < remainder else 0)
            chunk_scene_objs = scenes[start:end]
            chunks.append({
                "chunk_id": i + 1,
                "chunk_scenes": [scene["scene_id"] for scene in chunk_scene_objs],
                "characters": list(set(char for scene in chunk_scene_objs for char in scene["characters"])),
                "scene_texts": [scene["scene_text"] for scene in chunk_scene_objs]
            })
            start = end
    return chunks

def character_profiler(state: State) -> State:
    """Extracts and profiles characters from screenplay using LLM."""

    print("=> Profiling characters...")

    if not state.raw_script:
        print("Warning: No raw script found in state.")
        return state

    # Split into scenes
    scenes = simple_scene_splitter(state.raw_script)
    
    # Debug print
    print("Debug - Scene data:", json.dumps(scenes, indent=2))

    # Gather all unique character names
    all_chars = set()
    for scene in scenes:
        all_chars.update(scene.get("characters", []))

    print(f"=> Found {len(all_chars)} characters: {sorted(all_chars)}")

    characters = {}

    for char in all_chars:
        char_lines = []
        char_actions = []

        for scene in scenes:
            scene_text = scene.get("scene_text", "")

            # Dialogues: lines like ALICE: Hello!
            dialogue_matches = re.finditer(
                rf"^{re.escape(char)}:\s*(.+)$", scene_text, re.MULTILINE
            )
            char_lines.extend(match.group(1).strip() for match in dialogue_matches)

            # Actions: any line with the character's name not starting their dialogue
            action_lines = [
                line.strip() for line in scene_text.split('\n')
                if char in line and not line.strip().startswith(f"{char}:")
            ]
            char_actions.extend(action_lines)

        # Skip empty characters (false positives)
        if not char_lines and not char_actions:
            continue

        # Modified LLM prompt
        prompt = f"""Analyze this character's speaking style and emotional arc based on their lines and actions.

Character: {char}

Sample Dialogues:
{chr(10).join(char_lines)}

Sample Actions:
{chr(10).join(char_actions)}

Provide a JSON response with this exact format:
{{
    "speaking_style": "Brief description of how they speak",
    "emotional_arc": "Brief description of their emotional journey",
    "consistency_notes": "Any notes about character consistency"
}}"""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            
            # Add error handling for JSON parsing
            try:
                profile = json.loads(response.content)
            except json.JSONDecodeError as json_err:
                print(f"Warning: Failed to parse JSON for '{char}'. Using default profile.")
                print(f"Response was: {response.content}")
                profile = {
                    "speaking_style": "Unknown",
                    "emotional_arc": "Unknown",
                    "consistency_notes": "Failed to parse character profile"
                }

            characters[char] = Character(
                name=char,
                speaking_style=profile.get("speaking_style", "Unknown"),
                emotional_arc=profile.get("emotional_arc", "Unknown")
            )

        except Exception as e:
            print(f"Warning: Failed to profile '{char}': {str(e)}")
            characters[char] = Character(name=char)

    state.characters = characters
    return state


def scene_parser(state: State) -> State:
    """Parse raw script into structured scene chunks using the chunking logic"""
    print(" Parsing script into scene chunks...")
    
    # Use the integrated chunk maker logic
    state.scene_chunks = chunk_scenes_with_text(state.raw_script, threshold=10)
    
    # Convert chunks to Scene objects
    scenes = []
    
    for chunk in state.scene_chunks:
        for i, scene_text in enumerate(chunk["scene_texts"]):
            scene_id = chunk["chunk_scenes"][i]
            
            # Parse dialogues and actions from scene text
            dialogues = []
            actions = []
            characters = []
            
            lines = scene_text.split('\n')
            current_character = None
            current_dialogue = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for character dialogue with improved regex
                char_match = re.match(r'^([A-Z][A-Z0-9_ ]{1,})(?:\s*:)?\s*(.*)', line)
                if char_match:
                    # Save previous character's dialogue if any
                    if current_character and current_dialogue:
                        dialogues.append({"character": current_character, "line": " ".join(current_dialogue)})
                        if current_character not in characters:
                            characters.append(current_character)
                        current_dialogue = []
                    
                    char_name = char_match.group(1).strip()
                    dialogue = char_match.group(2).strip()
                    
                    if char_name not in characters:
                        characters.append(char_name)
                    
                    if dialogue:
                        current_dialogue.append(dialogue)
                    current_character = char_name
                elif current_character and line and not line.startswith('(') and not line.isupper() and not re.match(r'^[A-Z\s]+$', line):
                    # Continuation of dialogue
                    current_dialogue.append(line)
                elif line.startswith('(') and current_character:
                    # Parenthetical action - skip but keep current character
                    continue
                elif re.match(r'^[A-Z\s]+$', line) and len(line) > 2:
                    # Another character name in all caps
                    if current_character and current_dialogue:
                        dialogues.append({"character": current_character, "line": " ".join(current_dialogue)})
                        if current_character not in characters:
                            characters.append(current_character)
                        current_dialogue = []
                    current_character = line.strip()
                    if current_character not in characters:
                        characters.append(current_character)
                elif not line.isupper() and not line.startswith('('):
                    # Action line
                    actions.append(line)
            
            # Don't forget the last character's dialogue
            if current_character and current_dialogue:
                dialogues.append({"character": current_character, "line": " ".join(current_dialogue)})
                if current_character not in characters:
                    characters.append(current_character)
            
            scene = Scene(
                characters=characters,
                dialogues=dialogues,
                actions=actions,
                scene_id=scene_id
            )
            scenes.append(scene)
    
    state.scenes = scenes
    print(f"Parsed {len(scenes)} scenes into {len(state.scene_chunks)} chunks")
    return state


# def tone_analyzer(state: State) -> State:
#     """Analyze emotional tone and atmosphere for each scene"""
#     print("Analyzing scene tones and emotional atmosphere...")

#     tone_options = [
#         "tension", "humor", "sadness", "mystery", "romance", "conflict", "joy",
#         "fear", "anger", "neutral", "suspense", "melancholy", "hope"
#     ]
#     MAX_RETRIES = 1

#     for scene in state.scenes:
#         scene_content = ""

#         if scene.actions:
#             scene_content += "Actions/Description:\n" + "\n".join(scene.actions) + "\n\n"

#         if scene.dialogues:
#             scene_content += "Dialogue:\n"
#             for dialogue in scene.dialogues:
#                 scene_content += f"{dialogue['character']}: {dialogue['line']}\n"

#         # Skip very short scenes
#         if len(scene_content.strip()) < 50:
#             scene.tone_tags = ["neutral"]
#             continue

#         prompt = f"""You are an expert screenplay analyst specializing in emotional tone and atmosphere. 

# Analyze the emotional tone and atmosphere of this scene. Consider:
# - The emotional weight of the dialogue
# - The subtext and underlying tensions
# - The setting and action descriptions
# - Character interactions and dynamics
# - The overall mood and feeling the scene creates

# Available tone tags: {', '.join(tone_options)}

# Scene to analyze:
# {scene_content}

# Select 1-3 most appropriate tone tags that capture the essence of this scene. Return as a JSON array:
# ["primary_tone", "secondary_tone"]

# Consider both obvious emotions and subtle undertones. If the scene is neutral or unclear, return ["neutral"]."""

#         try:
#             success = False
#             for attempt in range(MAX_RETRIES):
#                 try:
#                     response = llm.invoke([HumanMessage(content=prompt)])
#                     tags = json.loads(response.content)
#                     success = True
#                     break
#                 except json.JSONDecodeError:
#                     # Try to extract JSON from partial content
#                     json_match = re.search(r'\[.*?\]', response.content, re.DOTALL)
#                     if json_match:
#                         try:
#                             tags = json.loads(json_match.group())
#                             success = True
#                             break
#                         except json.JSONDecodeError:
#                             continue
#                     else:
#                         continue

#             if not success:
#                 # Fallback: match tones from plain text
#                 tone_words = [tone for tone in tone_options if tone.lower() in response.content.lower()]
#                 tags = tone_words if tone_words else ["neutral"]

#             # Validate tags
#             if isinstance(tags, list):
#                 scene.tone_tags = [tag for tag in tags if tag in tone_options] or ["neutral"]
#             elif isinstance(tags, str):
#                 scene.tone_tags = [tags] if tags in tone_options else ["neutral"]
#             else:
#                 scene.tone_tags = ["neutral"]

#         except Exception as e:
#             print(f"Error analyzing tone for scene {scene.scene_id}: {e}")
#             scene.tone_tags = ["neutral"]

#     return state
def tone_analyzer(state: State) -> State:
    """Analyze emotional tone and atmosphere for each scene"""
    
    print("Analyzing scene tones and emotional atmosphere...")

    tone_options = [
        "tension", "humor", "sadness", "mystery", "romance", "conflict", "joy",
        "fear", "anger", "neutral", "suspense", "melancholy", "hope"
    ]
    MAX_RETRIES = 3  # Increased retries for more stability

    # Track processed scenes to avoid infinite loops
    processed_scenes = set()
    
    for scene in state.scenes:
        if scene.scene_id in processed_scenes:
            continue
            
        processed_scenes.add(scene.scene_id)
        
        # Skip if scene already has valid tone tags
        if hasattr(scene, 'tone_tags') and isinstance(scene.tone_tags, list) and scene.tone_tags:
            print(f"Scene {scene.scene_id} already has tone tags: {scene.tone_tags}")
            continue

        scene_content = ""
        if scene.actions:
            scene_content += "Actions:\n" + "\n".join(scene.actions) + "\n\n"
        if scene.dialogues:
            scene_content += "Dialogue:\n"
            for dialogue in scene.dialogues:
                scene_content += f"{dialogue['character']}: {dialogue['line']}\n"

        # Skip very short scenes
        if len(scene_content.strip()) < 50:
            scene.tone_tags = ["neutral"]
            print(f"Scene {scene.scene_id} too short - marked as neutral")
            continue

        prompt = f"""Analyze this scene's emotional tone. Select 1-3 tags from: {', '.join(tone_options)}
Scene content:
{scene_content}

Return ONLY a JSON array of tone tags. Example: ["tension", "fear"]"""

        retry_count = 0
        success = False
        
        while retry_count < MAX_RETRIES and not success:
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                response_text = response.content.strip()
                
                # Try to extract JSON array
                match = re.search(r'\[.*?\]', response_text)
                if match:
                    try:
                        tags = json.loads(match.group())
                        if isinstance(tags, list):
                            # Validate tags
                            valid_tags = [tag for tag in tags if tag in tone_options]
                            if valid_tags:
                                scene.tone_tags = valid_tags[:3]  # Limit to 3 tags
                                success = True
                                print(f"Scene {scene.scene_id} analyzed: {scene.tone_tags}")
                                break
                    except json.JSONDecodeError:
                        pass
                
                retry_count += 1
                if not success:
                    print(f"Retry {retry_count} for scene {scene.scene_id}")
                    
            except Exception as e:
                print(f"Error analyzing scene {scene.scene_id}: {str(e)}")
                retry_count += 1

        # Fallback if all retries failed
        if not success:
            scene.tone_tags = ["neutral"]
            print(f"Failed to analyze scene {scene.scene_id} after {MAX_RETRIES} attempts - marked as neutral")

    return state
def dialogue_enhancer(state: State) -> State:
    """
    Enhance dialogue lines using LLM while preserving character voice and intent.
    Works with the existing State structure containing scenes.
    """
    
    # Import LLM - assuming it's globally available or you can import it here
    from langchain_groq import ChatGroq
    llm = ChatGroq(temperature=0.7, model_name="llama3-70b-8192")  # Adjust model name as needed
    
    def _create_enhancement_prompt(dialogues: List[Dict[str, str]], scene_characters: List[str], tone_tags: List[str] = None) -> str:
        """Create a prompt for enhancing a list of dialogues with character context."""
        
        dialogues_text = []
        for i, dialogue_dict in enumerate(dialogues, 1):
            for character, line in dialogue_dict.items():
                dialogues_text.append(f"{i}. {character}: \"{line}\"")
        
        # Add character context if available
        character_context = ""
        if scene_characters:
            character_context = f"\nCHARACTERS IN SCENE: {', '.join(scene_characters)}"
        
        # Add tone context if available
        tone_context = ""
        if tone_tags:
            tone_context = f"\nSCENE TONE: {', '.join(tone_tags)}"
        
        prompt = f"""You are a professional dialogue enhancer for screenwriting and storytelling. Your task is to enhance the following dialogue lines while strictly following these guidelines:

DIALOGUE TO ENHANCE:
{chr(10).join(dialogues_text)}{character_context}{tone_context}

ENHANCEMENT RULES:
1. PRESERVE CHARACTER NAMES: Keep each character name exactly as provided
2. PRESERVE INTENT: The core meaning and purpose of each line must remain unchanged
3. PRESERVE STRUCTURE: Return the same number of lines in the same order
4. CHARACTER VOICE: Maintain each character's unique voice (sarcastic, soft, cold, anxious, etc.)
5. EMOTIONAL ENHANCEMENT: For emotional lines (sad, romantic, intense), increase impact subtly - NO melodrama
6. HUMOR ENHANCEMENT: For humorous lines, improve wit or timing without changing the joke's intent
7. NATURAL PACING: Ensure dialogue flows naturally - don't overwrite or make lines too long
8. SUBTLE SUBTEXT: Add emotional depth subtly - don't make subtext obvious or heavy-handed
9. SCENE INTEGRATION: Enhanced lines should blend seamlessly into any existing scene
10. TONE CONSISTENCY: Match the overall scene tone while enhancing individual lines

RESPONSE FORMAT:
Return ONLY a valid JSON array in this exact format:
[
    {{"Character Name": "Enhanced dialogue line"}},
    {{"Character Name": "Enhanced dialogue line"}},
    ...
]

IMPORTANT: 
- Return ONLY the JSON array, no additional text or explanation
- Each dialogue should be enhanced but feel natural and authentic
- Maintain the original emotional tone while making it more impactful
- Keep character personality consistent throughout
- Ensure enhanced dialogue sounds like something the character would actually say
- Consider the scene's overall tone when enhancing individual lines

JSON Response:"""
        
        return prompt
    
    def _parse_llm_response(response: str, original_dialogues: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Parse LLM response and handle potential errors."""
        try:
            # Clean the response - remove any extra text before/after JSON
            response = response.strip()
            
            # Find JSON array in response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON array found in response")
            
            json_str = response[start_idx:end_idx]
            enhanced_dialogues = json.loads(json_str)
            
            # Validate structure
            if not isinstance(enhanced_dialogues, list):
                raise ValueError("Response is not a list")
            
            if len(enhanced_dialogues) != len(original_dialogues):
                raise ValueError(f"Length mismatch: expected {len(original_dialogues)}, got {len(enhanced_dialogues)}")
            
            # Validate each dialogue has correct character names
            for i, (original, enhanced) in enumerate(zip(original_dialogues, enhanced_dialogues)):
                if not isinstance(enhanced, dict):
                    raise ValueError(f"Dialogue {i+1} is not a dictionary")
                
                original_chars = set(original.keys())
                enhanced_chars = set(enhanced.keys())
                
                if original_chars != enhanced_chars:
                    raise ValueError(f"Character name mismatch in dialogue {i+1}")
            
            return enhanced_dialogues
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response was: {response}")
            # Return original dialogues if parsing fails
            return original_dialogues
    
    def _enhance_scene_dialogues(scene: Scene) -> List[Dict[str, str]]:
        """Enhance dialogues for a single scene using LLM."""
        if not scene.dialogues:
            return scene.dialogues
        
        # Create prompt with scene context
        prompt = _create_enhancement_prompt(
            scene.dialogues, 
            scene.characters, 
            scene.tone_tags
        )
        
        # Invoke LLM
        try:
            response = llm.invoke([HumanMessage(content=prompt)]).content
            
            # Parse and validate response
            enhanced_dialogues = _parse_llm_response(response, scene.dialogues)
            return enhanced_dialogues
            
        except Exception as e:
            print(f"Error during LLM invocation: {e}")
            # Return original dialogues if LLM call fails
            return scene.dialogues
    
    # Process each scene if scenes exist
    if state.scenes:
        enhanced_scenes = []
        
        for scene in state.scenes:
            # Create a copy of the scene
            enhanced_scene = Scene(
                characters=scene.characters.copy() if scene.characters else [],
                dialogues=_enhance_scene_dialogues(scene),
                actions=scene.actions.copy() if scene.actions else [],
                tone_tags=scene.tone_tags.copy() if scene.tone_tags else None
            )
            enhanced_scenes.append(enhanced_scene)
        
        # Return updated state
        return State(
            raw_script=state.raw_script,
            characters=state.characters,
            scenes=enhanced_scenes,
            current_round=state.current_round,
            max_rounds=state.max_rounds,
            issues_found=state.issues_found,
            final_script=state.final_script
        )
    
    # If no scenes, return state unchanged
    return state

def conflict_resolver(state: State) -> State:
    """Identify and resolve character consistency issues"""
    print(" Resolving conflicts...")
    
    if not state.scenes or not state.characters:
        return state
    
    issues = []
    
    for scene in state.scenes:
        for dialogue in scene.dialogues:
            char = dialogue["character"]
            if char in state.characters:
                character = state.characters[char]
                
                # Check if dialogue matches character profile
                prompt = f"""
                Character Profile:
                Name: {character.name}
                Speaking Style: {character.speaking_style}
                Emotional Arc: {character.emotional_arc}
                
                Dialogue to evaluate: "{dialogue['line']}"
                
                Does this dialogue match the character's profile? 
                Return JSON response:
                {{
                    "consistent": boolean,
                    "issue": "description of inconsistency if any",
                    "suggestion": "improved dialogue if inconsistent"
                }}
                """
                
                response = llm.invoke([HumanMessage(content=prompt)])
                
                try:
                    evaluation = json.loads(response.content)
                    if not evaluation.get("consistent", True):
                        issues.append(f"Scene {scene.scene_id}: {evaluation['issue']}")
                        dialogue["line"] = evaluation.get("suggestion", dialogue["line"])
                except:
                    continue
    
    state.issues_found = issues
    return state

def rewrite_coordinator(state: State) -> State:
    """Coordinate the rewrite process"""
    print(f" Coordinating rewrite round {state.current_round + 1}")
    state.current_round += 1
    return state

def script_formatter(state: State) -> State:
    """Format the final script"""
    print(" Formatting final script...")
    
    formatted_script = ""
    
    for scene in state.scenes:
        formatted_script += f"\n\n"
        # Add tone tags
        if scene.tone_tags:
            formatted_script += f"[Tone: {', '.join(scene.tone_tags)}]\n\n"
        
        # Interleave actions and dialogues in order
        for action in scene.actions:
            if action.strip():
                formatted_script += f"{action}\n\n"
        
        for dialogue in scene.dialogues:
            formatted_script += f"{dialogue['character']}: {dialogue['line']}\n\n"
        
        formatted_script += "\n"
    
    state.final_script = formatted_script
    return state

def should_continue(state: State) -> str:
    """Decide whether to continue enhancement or finish"""
    if state.issues_found and state.current_round < state.max_rounds:
        return "continue"
    return "finish"

# Build the graph
def create_screenplay_pipeline():
    workflow = StateGraph(State)
    # Add nodes
    workflow.add_node("character_profiler", character_profiler)
    workflow.add_node("scene_parser", scene_parser)
    workflow.add_node("tone_analyzer", tone_analyzer)
    workflow.add_node("dialogue_enhancer", dialogue_enhancer)
    workflow.add_node("conflict_resolver", conflict_resolver)
    workflow.add_node("rewrite_coordinator", rewrite_coordinator)
    workflow.add_node("script_formatter", script_formatter)
    # Set entry point
    workflow.set_entry_point("character_profiler")
    # Add edges
    workflow.add_edge("character_profiler", "scene_parser")
    workflow.add_edge("scene_parser", "tone_analyzer")
    workflow.add_edge("tone_analyzer", "dialogue_enhancer")
    workflow.add_edge("dialogue_enhancer", "conflict_resolver")
    # Helper function to enforce max of 3 loops
    def should_continue_with_limit(state):
        # Use state.rewrite_count to track iterations (initialize if missing)
        if not hasattr(state, 'rewrite_count'):
            state.rewrite_count = 0
        if state.rewrite_count < 3 and should_continue(state) == "continue":
            state.rewrite_count += 1
            return "continue"
        return "finish"
    # Use the new conditional edge function
    workflow.add_conditional_edges(
        "conflict_resolver",
        should_continue_with_limit,
        {
            "continue": "rewrite_coordinator",
            "finish": "script_formatter"
        }
    )
    workflow.add_edge("rewrite_coordinator", "tone_analyzer")
    workflow.add_edge("script_formatter", END)
    return workflow.compile()

# Usage functions
def enhance_screenplay(raw_script: str) -> str:
    """Main function to enhance a screenplay"""
    pipeline = create_screenplay_pipeline()
    
    initial_state = State(raw_script=raw_script)
    result = pipeline.invoke(initial_state)
    
    print("\n Enhancement complete!")
    print(f"Processed {len(result['scenes'])} scenes")
    print(f"Found {len(result['characters'])} characters")
    print(f"Completed {result['current_round']} enhancement rounds")
    
    return result["final_script"]

def handle_file(file_path: str) -> str:
    """Handle different file types and extract text"""
    _, ext = os.path.splitext(file_path)
    if ext == ".txt":
        return extract_txt_text(file_path)
    elif ext == ".docx":
        return extract_docx_text(file_path)
    elif ext == ".pdf":
        return extract_pdf_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# Example usage
if __name__ == "__main__":
    sample_script = """
INT. COFFEE SHOP - DAY

ALICE sits at a corner table, nervously checking her phone.
JACK enters, scanning the room.
ALICE: Where is he? He's twenty minutes late.

BOB rushes in, looking frazzled.
JACK: (to himself) I hope I'm not too late.

BOB: Sorry! Traffic was crazy.
ALICE: I was starting to think you stood me up.
BOB: Never. You're too important to me.
JACK: (smiling) I just got caught up with work.
INT. RESTAURANT - NIGHT

ALICE and BOB sit across from each other, the mood more relaxed.

ALICE: I'm glad we're finally doing this.
BOB: Me too. I've been wanting to ask you out for months.
JACK: (to himself) I hope this goes well.
ALICE: Really? I had no idea.
BOB: I'm not good at showing my feelings, I guess.
"""
    
    # For file input, uncomment and modify:
    # script_text = handle_file("data/script.pdf")
    # enhanced = enhance_screenplay(script_text)
    
    enhanced = enhance_screenplay(sample_script)
    print("\n" + "="*50)
    print("ENHANCED SCRIPT:")
    print("="*50)
    print(enhanced)