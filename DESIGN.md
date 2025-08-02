# JSON Extraction Backend Design

## Problem Statement
Extract structured JSON data from unstructured text using predefined JSON schemas for validation.

## Current Solution: LLM-Based Extraction with Retry Logic

### Approach
```
Text Input + JSON Schema → LLM → JSON Output → Schema Validation → Success/Retry
```

### Implementation
1. **Input**: Unstructured text + target JSON schema
2. **LLM Processing**: Send both to Google Gemini 2.5 Flash  
3. **Validation**: Validate output against JSON Schema Draft 7
4. **Retry Logic**: If validation fails, retry up to 3 times
5. **Output**: Validated JSON + metadata (attempts, validation status)

### Core Components

#### `BaseExtractor` (Template Method Pattern)
```python
def extract(text: str, schema: dict, max_retries: int = 3) -> ExtractResponse:
    for attempt in range(1, max_retries + 1):
        json_data = self._call_llm(text, schema)  # Abstract method
        is_valid, errors = self._validate_json(json_data, schema)
        if is_valid:
            return ExtractResponse(data=json_data, valid=True, attempts=attempt)
    return ExtractResponse(valid=False, attempts=max_retries, error="Max retries exceeded")
```

#### `GeminiExtractor` (Concrete Implementation)
- Formats prompt with text + schema requirements
- Calls Google Gemini API synchronously
- Parses JSON response with error handling

### Pros of Current Solution

✅ **Simple & Reliable**: Straightforward LLM call with validation  
✅ **Schema Enforcement**: Guarantees output matches required structure  
✅ **Automatic Retry**: Handles transient LLM inconsistencies  
✅ **Extensible**: Easy to add new LLM providers via Protocol interface  

### Cons of Current Solution

❌ **No Learning from Failures**: LLM receives no context about previous failed attempts  
❌ **Blind Retries**: Each retry is identical - no improvement mechanism  
❌ **Schema Dependency**: Requires knowing the target schema upfront  
❌ **Limited Error Context**: Validation errors not fed back to improve extraction  

### Major Limitation: Missing Failure Context

**Current Flow:**
```
Attempt 1: LLM(text, schema) → Invalid JSON → Discard
Attempt 2: LLM(text, schema) → Invalid JSON → Discard  
Attempt 3: LLM(text, schema) → Success/Failure
```

**Missing:** Previous extraction results and validation errors are not passed back to the LLM, so it cannot learn from mistakes or refine its approach.


## Technology Choices

### Synchronous Processing
**Rationale**: Simple request-response for web interface  
**Trade-off**: Blocking calls but acceptable for 2-10 second response times  

### JSON Schema Draft 7 Validation
**Rationale**: Industry standard, comprehensive validation rules  
**Implementation**: `jsonschema` Python library  

## Performance Characteristics
- **Response Time**: 2-10 seconds (Gemini API dependent)