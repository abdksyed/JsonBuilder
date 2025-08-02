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

---

# Design Evolution: Iteration 2

## Problems Solved

### 1. Contextual Learning from Failures
**Previous Issue**: LLM received no feedback about validation failures  
**Solution**: Multi-turn conversation with error context

**New Flow:**
```
Attempt 1: LLM(text, schema) → Invalid JSON → Store error
Attempt 2: LLM(text, schema, previous_error) → Better JSON → Success
```

### 2. Intelligent Schema Selection
**Previous Issue**: Users must manually select appropriate schema  
**Solution**: AI-powered automatic schema detection

**Auto-Detection Flow:**
```
Text Input → Schema Summaries Analysis → Best Match Selection → JSON Extraction
```

### 3. Automated Metadata Generation  
**Previous Issue**: Manual title/summary creation for new schemas  
**Solution**: LLM-generated metadata with smart slug creation

## Advanced Architecture

### Multi-Agent System Design
- **AutoDetectAgent**: Handles schema selection and metadata generation
- **Enhanced ExtractorService**: Manages auto-extraction workflows
- **Conversation Management**: Multi-turn extraction with error context
- **Prompt Engineering**: Modular YAML configuration for prompt management

### Performance & Model Optimization

#### Model Configuration
- **Gemini Pro**: Used for JSON extraction (higher accuracy)
- **Gemini Flash**: Used for auto-detection and metadata generation (cost optimization)
- **Performance Metrics**: Real-time tracking of duration, cost, and token usage

#### Enhanced User Experience
- **Confirmation Workflow**: Schema selection → review → confirmation → extraction
- **Override Capability**: Manual schema selection option in auto-detect mode
- **Performance Dashboard**: Frontend display of metrics for transparency

### Frontend Enhancements

#### Auto-Detect Workflow
1. User enters text
2. AI selects best schema automatically
3. User reviews and confirms selection
4. JSON extraction with performance metrics

#### Statistics Display
- Extraction time and model information
- Token usage and cost tracking
- Success rates and attempt counts

## Quality Assurance & Monitoring

### Comprehensive Logging
```
2025-08-02 17:09:31,828 - INFO - Auto-detecting schema for text with 5 available schemas
2025-08-02 17:09:35,110 - INFO - Selected schema: resume-schema
2025-08-02 17:09:43,311 - INFO - Using model: gemini-2.5-pro
2025-08-02 17:10:01,754 - INFO - Total Statistics - Total Tokens: 5915, Total Price: 0.0136575
```

### Error Handling
- Graceful degradation for auto-detection failures
- Clear user feedback with recovery suggestions
- Robust validation at all system layers

## Performance Characteristics
- **Auto-Detection**: 5-10 seconds using Gemini Flash
- **JSON Extraction**: 20-30 seconds using Gemini Pro
- **Success Rate**: 95%+ first-attempt success for well-structured text