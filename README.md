# AI-powered Screenplay Enhancer

This project provides a pipeline for **analyzing and enhancing screenplays** using Large Language Models (LLMs).  
It extracts characters, splits scenes, analyzes tones, enhances dialogues, resolves inconsistencies, and produces a polished final script.

## Features
- Parse raw scripts into structured **scenes & characters**
- Extract scripts from `.txt`, `.docx`, `.pdf`
- Character profiling (speaking style & emotional arc)
- Scene tone analysis (tension, humor, mystery, etc.)
- Dialogue enhancement while preserving **character voice**
- Conflict resolution for character consistency
- Final script formatting with tone annotations

## Agentic AI Workflow
<img width="363" height="509" alt="image" src="https://github.com/user-attachments/assets/df1460d1-7536-4e1c-bb8a-2bc9362cb24d" />

## 📂 File Overview
- **`ai_screenplay.py`** → Core pipeline logic for screenplay enhancement
- **`file_handler.py`** → Utilities for extracting text from PDF, DOCX, TXT
- **`requirements.txt`** → Python dependencies
- **`.env`** → Store API keys (e.g., `GROQ_API_KEY=your_key_here`)

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/screenplay-enhancer.git
   cd screenplay-enhancer
