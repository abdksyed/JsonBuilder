# Implementation v0.1.0

## Overview
Schema-Aware JSON Extractor that transforms unstructured text into structured JSON using Google Gemini and JSON Schema validation.

## What's Implemented

### Core Features
- **JSON Extraction**: LLM-powered text-to-JSON conversion with schema validation
- **Schema Management**: File-based storage for custom JSON schemas  
- **Web Interface**: 3-tab UI for extraction, auto-detection, and schema creation
- **REST API**: Full CRUD endpoints for programmatic access
- **Retry Logic**: Auto-retry up to 3 times on validation failures

### Components
- `app/main.py` - FastAPI application with static file serving
- `app/api/routes.py` - REST endpoints for extraction and schema management
- `app/services/extractor.py` - Google Gemini integration with validation
- `app/repositories/schema_repo.py` - File-based schema storage
- `app/models.py` - Pydantic models for request/response validation
- `app/ui/templates/index.html` - Responsive web interface

### How It Works
1. User provides text and selects a JSON schema
2. System sends text + schema to Google Gemini 2.5 Flash
3. LLM returns structured JSON matching the schema
4. System validates output against JSON Schema Draft 7
5. If validation fails, retry with error context (max 3 attempts)

## Dependencies
- FastAPI, Pydantic, jsonschema, google-genai, uvicorn
- Frontend: Tailwind CSS + Axios (CDN)

## Setup
1. Set `GOOGLE_API_KEY` environment variable
2. Install: `uv pip install fastapi uvicorn pydantic jsonschema google-genai`
3. Run: `uv run run.py`