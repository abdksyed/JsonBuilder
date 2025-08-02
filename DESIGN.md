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

### Enhanced User Experience
- **Confirmation Workflow**: Schema selection → review → confirmation → extraction
- **Override Capability**: Manual schema selection option in auto-detect mode
- **Performance Dashboard**: Frontend display of metrics for transparency

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

---

# Design Evolution: Iteration 3 - Large Scale Architecture 

**Implemented Embedding Search. Hybrid Search, Metadata Filtering, BM25 Search, Re-ranking still to be implemented when we have more schemas.**

## Problem Statement: Scalability Bottlenecks

### Current Limitations at Scale
❌ **Linear Schema Comparison**: Auto-detection compares text against ALL schemas  
❌ **LLM Processing Overhead**: Every schema requires LLM evaluation  
❌ **Memory Constraints**: Loading 1000+ schemas impacts performance  
❌ **Response Latency**: Auto-detection time scales linearly with schema count  

## Solution: Hybrid Retrieval + Re-ranking Pipeline

### Two-Stage Architecture
```
Stage 1: Fast Retrieval (Vector (if needed, +BM25+Metadata)) → Top-K Candidates (5-10)
Stage 2: Intelligent Re-ranking (LLM/Re-ranker) → Best Schema Selection
```

### Stage 1: Hybrid Retrieval System

#### Vector Similarity Search (DONE)
```python
# Pre-computed schema embeddings for semantic matching
schema_embeddings = embed_text([schema.summary for schema in all_schemas])
query_embedding = embed_text(input_text)
similarity_scores = cosine_similarity(query_embedding, schema_embeddings)
top_semantic_matches = get_top_k(similarity_scores, k=20)
```

#### BM25 Keyword Search (TODO)
```python
# Traditional keyword matching for exact term overlap
bm25_index = build_bm25_index([schema.summary + schema.title for schema in schemas])
bm25_scores = bm25_index.get_scores(input_text)
top_keyword_matches = get_top_k(bm25_scores, k=20)
```

#### Metadata Filtering (TODO)
```python
# Fast pre-filtering based on domain, category, tags
def filter_by_metadata(text: str, schemas: List[Schema]) -> List[Schema]:
    detected_domain = detect_domain(text)  # e.g., "finance", "healthcare" 
    detected_type = detect_document_type(text)  # e.g., "resume", "invoice"
    
    return [s for s in schemas if 
           s.domain == detected_domain or 
           s.category == detected_type or
           any(tag in text.lower() for tag in s.tags)]
```

#### Hybrid Ranking Strategy (TODO)
```python
def hybrid_rank(embedding_results, bm25_results, metadata_results, weights=(0.5, 0.3, 0.2)):
    combined_scores = {}
    for schema_id in all_candidate_schemas:
        score = (weights[0] * embedding_scores.get(schema_id, 0) +
                weights[1] * bm25_scores.get(schema_id, 0) +  
                weights[2] * metadata_scores.get(schema_id, 0))
        combined_scores[schema_id] = score
    
    return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
```

### Stage 2: Intelligent Re-ranking

#### LLM-Based Re-ranking (Current)
```python
async def llm_rerank(text: str, candidates: List[SchemaMeta]) -> str:
    prompt = f"""
    Given this text: {text[:2000]}
    
    Choose the best schema from these {len(candidates)} candidates:
    {format_candidate_summaries(candidates)}
    
    Return only the schema ID of the best match.
    """
    return await llm_call(prompt, model="gemini-flash")
```

#### Dedicated Re-ranker Model (Future Enhancement)
```python
async def reranker_model(text: str, candidates: List[SchemaMeta]) -> str:
    # Cross-encoder or similar re-ranking model
    pairs = [(text, candidate.summary) for candidate in candidates]
    scores = await reranker.predict(pairs)
    best_idx = np.argmax(scores)
    return candidates[best_idx].id
```

## Performance Benefits

## Implementation Strategy

### Database Schema for Scale
```sql
-- Vector storage for embeddings
CREATE TABLE schema_embeddings (
    schema_id VARCHAR PRIMARY KEY,
    embedding VECTOR(1536),  -- OpenAI ada-002 dimensions
    created_at TIMESTAMP
);

-- BM25 index table
CREATE TABLE schema_keywords (
    schema_id VARCHAR,
    term VARCHAR,
    tf_idf_score FLOAT,
    INDEX(term, tf_idf_score)
);

-- Metadata for fast filtering
CREATE TABLE schema_metadata (
    schema_id VARCHAR PRIMARY KEY,
    domain VARCHAR,
    category VARCHAR,
    tags JSON,
    INDEX(domain), INDEX(category)
);
```

## Todo Enhancements Pipeline

### Phase 1: Building Metadata and Database for Dense Embedding and BM25 Search

### Phase 2: Re-ranker Integration
```python
class RerankerService:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.model = CrossEncoder(model_name)
    
    async def rank_candidates(self, query: str, candidates: List[str]) -> List[Tuple[str, float]]:
        pairs = [[query, candidate] for candidate in candidates]
        scores = self.model.predict(pairs)
        return sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
```