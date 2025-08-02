from typing import Protocol, Dict, Any, Optional, List
import json
import jsonschema
import yaml
import os
import logging
import time
import re
import numpy as np
from abc import ABC, abstractmethod
from app.models import SchemaMeta

# Configure logger with file handler
logger = logging.getLogger(__name__)

# Set up file handler if not already configured
if not logger.handlers:
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up file handler
    log_file = os.path.join(log_dir, 'services.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Set up console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)


def load_prompts() -> Dict[str, Any]:
    """Load prompts from YAML configuration file."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'prompts.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def calculate_usage_stats(response, model_config: Dict[str, Any], duration: float = None) -> Dict[str, Any]:
    """Calculate and log usage statistics from LLM response.
    
    Args:
        response: LLM response object with usage_metadata
        model_config: Model configuration containing pricing info
        duration: Optional duration in seconds
    
    Returns:
        Dictionary containing usage statistics
    """
    usage_stats = {}
    
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        usage = response.usage_metadata
        input_price = model_config.get('input_price', 0)
        output_price = model_config.get('output_price', 0)
        
        # Extract token counts
        input_tokens = getattr(usage, 'prompt_token_count', 0)
        output_tokens = getattr(usage, 'candidates_token_count', 0)
        thought_tokens = getattr(usage, 'thought_token_count', 0)
        total_tokens = getattr(usage, 'total_token_count', 0)
        
        # Calculate costs (prices are per million tokens)
        input_cost = (input_price * input_tokens) / 1e6
        output_cost = (output_price * output_tokens) / 1e6
        thought_cost = (output_price * thought_tokens) / 1e6
        total_cost = input_cost + output_cost + thought_cost
        
        # Log detailed statistics
        logger.info(f"Input Statistics - Input Tokens: {input_tokens}, Input Price: {input_cost}")
        logger.info(f"Output Statistics - Output Tokens: {output_tokens}, Output Price: {output_cost}")
        logger.info(f"Thought Statistics - Thought Tokens: {thought_tokens}, Thought Price: {thought_cost}")
        logger.info(f"Total Statistics - Total Tokens: {total_tokens}, Total Price: {total_cost}")
        
        # Build usage stats dictionary
        usage_stats = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thought_tokens": thought_tokens,
            "total_tokens": total_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "thought_cost": thought_cost,
            "total_cost": total_cost
        }
        
        if duration is not None:
            usage_stats["duration"] = duration
    
    return usage_stats


def create_llm_content(prompt: str, role: str = "user"):
    """Create LLM content object for Gemini API."""
    from google.genai.types import Part, Content
    return Content(parts=[Part.from_text(text=prompt)], role=role)


def clean_json_response(response_text: str) -> str:
    """Clean and extract JSON from LLM response."""
    return response_text.strip().replace("```json", "").replace("```", "")


def create_slug(text: str) -> str:
    """Create a URL-friendly slug from text."""
    slug = text.lower()
    slug = re.sub(r'[^a-z0-9-]', '-', slug)
    slug = re.sub(r'-+', '-', slug).strip('-')
    return slug


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class EmbeddingManager:
    """Manages schema embeddings for large-scale similarity search."""
    
    def __init__(self, extractor=None):
        self.extractor = extractor
        self.embeddings_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'data', 'schema_embeddings.npz'
        )
        self.schema_embeddings = None
        self.schema_ids = None
        self._ensure_data_directory()
    
    def _ensure_data_directory(self):
        """Ensure data directory exists."""
        data_dir = os.path.dirname(self.embeddings_file)
        os.makedirs(data_dir, exist_ok=True)
    
    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using text-embedding-004."""
        if not self.extractor:
            raise ValueError("Extractor client not available for embedding generation")
        
        try:
            response = self.extractor.client.models.embed_content(
                model="text-embedding-004",
                contents=text
            )
            return np.array(response.embeddings[0].values)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def build_embeddings(self, schemas: List[SchemaMeta]) -> None:
        """Build and save embeddings for all schemas."""
        logger.info(f"Building embeddings for {len(schemas)} schemas")
        
        embeddings = []
        schema_ids = []
        
        for schema in schemas:
            # Create text to embed: title + summary for richer context
            embed_text = f"{schema.title}. {schema.summary}"
            
            try:
                embedding = await self._generate_embedding(embed_text)
                embeddings.append(embedding)
                schema_ids.append(schema.id)
                logger.debug(f"Generated embedding for schema: {schema.id}")
            except Exception as e:
                logger.error(f"Failed to generate embedding for schema {schema.id}: {e}")
                continue
        
        if embeddings:
            # Save embeddings and IDs
            embeddings_matrix = np.array(embeddings)
            np.savez(self.embeddings_file, 
                    embeddings=embeddings_matrix, 
                    schema_ids=np.array(schema_ids))
            
            # Load into memory
            self.schema_embeddings = embeddings_matrix
            self.schema_ids = np.array(schema_ids)
            
            logger.info(f"Saved {len(embeddings)} embeddings to {self.embeddings_file}")
        else:
            logger.warning("No embeddings were generated")
    
    def load_embeddings(self) -> bool:
        """Load embeddings from file into memory."""
        if not os.path.exists(self.embeddings_file):
            logger.info("No embeddings file found, will need to build embeddings")
            return False
        
        try:
            data = np.load(self.embeddings_file)
            self.schema_embeddings = data['embeddings']
            self.schema_ids = data['schema_ids']
            logger.info(f"Loaded {len(self.schema_embeddings)} embeddings from file")
            return True
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return False
    
    async def find_best_schema(self, text: str, available_schemas: List[SchemaMeta], top_k: int = 1) -> List[str]:
        """Find best matching schemas using embedding similarity."""
        if self.schema_embeddings is None:
            logger.warning("No embeddings available, rebuilding...")
            await self.build_embeddings(available_schemas)
            if self.schema_embeddings is None:
                raise ValueError("Failed to build embeddings")
        
        # Generate embedding for input text
        input_embedding = await self._generate_embedding(text[:2000])  # Limit text length
        
        # Calculate similarities with all schema embeddings
        similarities = []
        available_ids = {schema.id for schema in available_schemas}
        
        for i, schema_id in enumerate(self.schema_ids):
            if schema_id in available_ids:  # Only consider available schemas
                similarity = cosine_similarity(input_embedding, self.schema_embeddings[i])
                similarities.append((similarity, schema_id))
        
        if not similarities:
            logger.warning("No matching schemas found in embeddings")
            return []
        
        # Sort by similarity and return top_k
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_schemas = [schema_id for _, schema_id in similarities[:top_k]]
        
        logger.info(f"Top embedding match: {top_schemas[0]} (similarity: {similarities[0][0]:.4f})")
        return top_schemas
    
    def needs_rebuild(self, current_schemas: List[SchemaMeta]) -> bool:
        """Check if embeddings need to be rebuilt based on current schemas."""
        if self.schema_embeddings is None:
            return True
        
        current_ids = {schema.id for schema in current_schemas}
        stored_ids = set(self.schema_ids) if self.schema_ids is not None else set()
        
        # Rebuild if schema sets don't match
        return current_ids != stored_ids


class ExtractorLLM(Protocol):
    """Protocol for LLM extractors that convert text to JSON according to a schema."""
    
    async def extract(self, text: str, schema: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """Extract JSON from text according to the provided schema.
        
        Returns:
            Dict with keys: data, valid, attempts, error (optional)
        """
        ...


class BaseExtractor(ABC):
    """Base class for LLM extractors."""
    
    def __init__(self):
        self.prompts = load_prompts()
    
    @abstractmethod
    async def _call_llm(self, text: str, schema: Dict[str, Any], attempt: int = 1, previous_error: Optional[str] = None) -> str:
        """Call the LLM and return raw JSON string."""
        pass

    @classmethod
    def _create_error_response(cls, error: str, max_retries: int) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "data": {},
            "valid": False,
            "attempts": max_retries,
            "error": error
        }
    
    def _parse_json_safely(self, raw_json: str) -> tuple[Dict[str, Any], Optional[str]]:
        """Parse JSON safely and return data or error message."""
        try:
            data = json.loads(raw_json)
            logger.debug(f"Successfully parsed JSON: {data}")
            return data, None
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON: {str(e)}"
            return {}, error_msg
    
    def _validate_schema(self, data: Dict[str, Any], schema_data: Dict[str, Any]) -> Optional[str]:
        """Validate data against schema and return error message if invalid."""
        try:
            jsonschema.validate(data, schema_data)
            return None
        except jsonschema.ValidationError as e:
            return f"Schema validation failed: {str(e)}"
    
    async def extract(self, text: str, schema: SchemaMeta, max_retries: int = 3) -> Dict[str, Any]:
        """Extract JSON with validation and repair loop."""
        logger.info(f"Starting JSON extraction with max_retries={max_retries}")
        logger.debug(f"Input text length: {len(text)} characters")
        logger.debug(f"Schema: {schema.title}")
        
        schema_data = schema.schema_data
        previous_error = None
        
        for attempt in range(1, max_retries + 1):
            logger.info(f"Extraction attempt {attempt}/{max_retries}")
            try:
                # Call the LLM with previous error context
                raw_json = await self._call_llm(text, schema_data, attempt, previous_error)
                logger.debug(f"LLM response: {raw_json.replace(chr(10), '\\n').replace(chr(9), '\\t')}")
                
                # Parse JSON safely
                data, parse_error = self._parse_json_safely(raw_json)
                if parse_error:
                    logger.warning(f"JSON parsing failed on attempt {attempt}: {parse_error}")
                    if attempt == max_retries:
                        logger.error(f"JSON parsing failed after {max_retries} attempts")
                        return self._create_error_response(parse_error, max_retries)
                    previous_error = parse_error
                    continue
                
                # Validate against schema
                validation_error = self._validate_schema(data, schema_data)
                if validation_error:
                    logger.warning(f"Schema validation failed on attempt {attempt}: {validation_error}")
                    if attempt == max_retries:
                        logger.error(f"Schema validation failed after {max_retries} attempts")
                        return self._create_error_response(validation_error, max_retries)
                    previous_error = validation_error
                    continue
                
                # Success case
                logger.info(f"Successfully extracted and validated JSON on attempt {attempt}")
                return {
                    "data": data,
                    "valid": True,
                    "attempts": attempt
                }
                    
            except Exception as e:
                error_msg = f"Extraction failed: {str(e)}"
                logger.error(f"Unexpected error on attempt {attempt}: {error_msg}")
                if attempt == max_retries:
                    logger.error(f"Extraction failed after {max_retries} attempts")
                    return self._create_error_response(error_msg, max_retries)
                # Don't set previous_error for internal errors (rate limits, etc.)
                previous_error = ""
                continue
        
        logger.error("Max retries exceeded")
        return self._create_error_response("Max retries exceeded", max_retries)


class GeminiExtractor(BaseExtractor):
    """Gemini-based JSON extractor with multi-turn chat support."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        from google import genai
        
        # Configure API key
        api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            logger.error("Google API key is required but not provided")
            raise ValueError("Google API key is required")
            
        logger.info("Initializing GeminiExtractor")
        self.client = genai.Client(api_key=api_key)
        self.conversation_history: List = []
    
    def _build_conversation_content(self, text: str, schema: Dict[str, Any], attempt: int, previous_error: Optional[str]):
        """Build conversation content for the current attempt."""
        prompts = self.prompts['prompts']
        
        if attempt == 1:
            # Initialize conversation with first prompt
            self.conversation_history = []
            schema_str = json.dumps(schema, indent=2)
            initial_prompt = prompts['initial_extraction'].format(
                schema=schema_str,
                text=text
            )
            self.conversation_history.append(create_llm_content(initial_prompt))
        else:
            # Add retry prompt with error context
            if previous_error:
                error_message = prompts['retry_with_error'].format(error=previous_error)
            else:
                error_message = prompts['retry_generic'].format(attempt=attempt)
            
            self.conversation_history.append(create_llm_content(error_message))
    
    async def _call_llm(self, text: str, schema: Dict[str, Any], attempt: int = 1, previous_error: Optional[str] = None) -> str:
        """Call Gemini to extract JSON using multi-turn conversation."""
        # Build conversation content
        self._build_conversation_content(text, schema, attempt, previous_error)
        
        logger.info(f"Sending request to Gemini (attempt {attempt})")
        logger.debug(f"Conversation history length: {len(self.conversation_history)}")
        
        # Generate content using conversation history
        model_config = self.prompts['model_config']['extraction']
        model_name = model_config['model_name']
        logger.debug(f"Using model: {model_name}")
        
        response = self.client.models.generate_content(
            model=model_name,
            contents=self.conversation_history
        )
        
        # Calculate and log usage statistics
        calculate_usage_stats(response, model_config)
        
        # Extract and clean the generated text
        generated_text = clean_json_response(response.text)
        logger.debug(f"Generated text length: {len(generated_text)} characters")
        
        # Add the model's response to conversation history
        if response.candidates:
            candidate_content = response.candidates[0].content
            self.conversation_history.append(candidate_content)
        
        return generated_text
    
    def reset_conversation(self):
        """Reset the conversation history for a new extraction task."""
        logger.debug("Resetting conversation history")
        self.conversation_history = []
    
    def get_conversation_history(self) -> List:
        """Get the current conversation history for debugging."""
        logger.debug(f"Returning conversation history with {len(self.conversation_history)} messages")
        return self.conversation_history


class AutoDetectAgent:
    """Agent for automatically detecting the best schema for given text."""
    
    def __init__(self, extractor: Optional[GeminiExtractor] = None):
        logger.info("Initializing AutoDetectAgent")
        self.extractor = extractor or GeminiExtractor(os.getenv('GOOGLE_API_KEY'))
        self.prompts = load_prompts()
        self.embedding_manager = EmbeddingManager(self.extractor)
        self.embedding_manager.load_embeddings()
    
    def _format_schemas_for_selection(self, schemas: List[SchemaMeta]) -> str:
        """Format schemas for LLM selection prompt."""
        schema_summaries = []
        for schema in schemas:
            schema_summaries.append(f"ID: {schema.id}\nTitle: {schema.title}\nSummary: {schema.summary}")
        return "\n\n".join(schema_summaries)
    
    def _validate_schema_selection(self, selected_id: str, available_schemas: List[SchemaMeta]) -> str:
        """Validate and return the best schema ID."""
        valid_ids = [schema.id for schema in available_schemas]
        if selected_id not in valid_ids:
            logger.warning(f"Invalid schema ID selected: {selected_id}, using first available")
            return valid_ids[0] if valid_ids else None
        return selected_id
    
    async def select_schema_embedding(self, text: str, available_schemas: List[SchemaMeta]) -> Dict[str, Any]:
        """Select the best schema using embedding similarity."""
        start_time = time.time()
        logger.info(f"Auto-detecting schema using embeddings for text with {len(available_schemas)} available schemas")
        
        # Check if embeddings need rebuilding
        if self.embedding_manager.needs_rebuild(available_schemas):
            logger.info("Rebuilding schema embeddings...")
            await self.embedding_manager.build_embeddings(available_schemas)
        
        try:
            # Find best matching schema using embeddings
            best_schema_ids = await self.embedding_manager.find_best_schema(text, available_schemas)
            
            if not best_schema_ids:
                # Fallback to first available schema
                logger.warning("No embedding matches found, using first available schema")
                selected_schema_id = available_schemas[0].id if available_schemas else None
            else:
                selected_schema_id = best_schema_ids[0]
            
            # Find the selected schema object
            selected_schema = next((s for s in available_schemas if s.id == selected_schema_id), None)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Create usage stats (embeddings are much faster and cheaper)
            usage_stats = {
                "method": "embedding",
                "duration": duration,
                "total_cost": 0.001,  # Approximate cost for embedding generation
                "total_tokens": 0  # Embedding doesn't use tokens like LLM
            }
            
            logger.info(f"Selected schema via embedding: {selected_schema_id}")
            
            return {
                "schema_id": selected_schema_id,
                "schema": selected_schema,
                "usage_stats": usage_stats
            }
            
        except Exception as e:
            logger.error(f"Embedding-based selection failed: {e}")
            # Fallback to LLM-based selection
            logger.info("Falling back to LLM-based selection")
            return await self.select_schema_llm(text, available_schemas)
    
    async def select_schema_llm(self, text: str, available_schemas: List[SchemaMeta]) -> Dict[str, Any]:
        """Select the best schema using LLM analysis."""
        start_time = time.time()
        logger.info(f"Auto-detecting schema using LLM for text with {len(available_schemas)} available schemas")
        
        # Format schemas and create prompt
        schemas_text = self._format_schemas_for_selection(available_schemas)
        prompt = self.prompts['prompts']['schema_selection'].format(
            text=text[:2000],  # Limit text length for prompt
            schemas=schemas_text
        )
        
        logger.debug("Sending schema selection request to LLM")
        model_config = self.prompts['model_config']['auto_detect']
        
        response = self.extractor.client.models.generate_content(
            model=model_config['model_name'],
            contents=[create_llm_content(prompt)]
        )
        
        # Calculate timing and usage statistics
        duration = time.time() - start_time
        usage_stats = calculate_usage_stats(response, model_config, duration)
        usage_stats["method"] = "llm"
        
        # Process and validate selection
        selected_schema_id = response.text.strip()
        logger.info(f"Selected schema via LLM: {selected_schema_id}")
        
        validated_id = self._validate_schema_selection(selected_schema_id, available_schemas)
        selected_schema = next((s for s in available_schemas if s.id == validated_id), None)
        
        return {
            "schema_id": validated_id,
            "schema": selected_schema,
            "usage_stats": usage_stats
        }
    
    async def select_schema(self, text: str, available_schemas: List[SchemaMeta], method: str = "embedding") -> Dict[str, Any]:
        """Select the best schema for the given text using specified method."""
        if method == "embedding":
            return await self.select_schema_embedding(text, available_schemas)
        elif method == "llm":
            return await self.select_schema_llm(text, available_schemas)
        else:
            raise ValueError(f"Unknown selection method: {method}. Use 'embedding' or 'llm'")
    
    async def _generate_single_metadata(self, prompt_key: str, schema_str: str) -> str:
        """Generate a single piece of metadata using LLM."""
        prompt = self.prompts['prompts'][prompt_key].format(schema=schema_str)
        model_config = self.prompts['model_config']['metadata_generation']
        
        response = self.extractor.client.models.generate_content(
            model=model_config['model_name'],
            contents=[create_llm_content(prompt)]
        )
        
        return response.text.strip().strip('"\'')
    
    async def generate_schema_metadata(self, schema_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate title, summary, and slug for a schema."""
        logger.info("Generating schema metadata")
        schema_str = json.dumps(schema_data, indent=2)
        
        # Generate title and summary in parallel operations
        title = await self._generate_single_metadata('generate_title', schema_str)
        summary = await self._generate_single_metadata('generate_summary', schema_str)
        
        # Generate slug based on title and summary
        slug_prompt = self.prompts['prompts']['generate_slug'].format(
            title=title,
            summary=summary
        )
        model_config = self.prompts['model_config']['metadata_generation']
        
        slug_response = self.extractor.client.models.generate_content(
            model=model_config['model_name'],
            contents=[create_llm_content(slug_prompt)]
        )
        
        # Clean and format slug
        slug = create_slug(slug_response.text.strip().strip('"\''))
        
        logger.info(f"Generated metadata - Title: {title}, Slug: {slug}")
        
        return {
            "title": title,
            "summary": summary,
            "slug": slug
        }


class ExtractorService:
    """Service that manages JSON extraction using different LLM backends."""
    
    def __init__(self, extractor: Optional[ExtractorLLM] = None):
        logger.info("Initializing ExtractorService")
        self.extractor = extractor or GeminiExtractor(os.getenv('GOOGLE_API_KEY'))
        self.auto_detect_agent = AutoDetectAgent(
            self.extractor if isinstance(self.extractor, GeminiExtractor) else None
        )

    def _create_error_result(self, start_time: float, selection_stats: Dict = None) -> Dict[str, Any]:
        """Create error result for auto-extraction."""
        return {
            "schema_id": None,
            "schema_title": None,
            "schema_summary": None,
            "data": {},
            "valid": False,
            "attempts": 0,
            "error": "No valid schema available for auto-detection",
            "stats": {
                "total_duration": time.time() - start_time,
                "selection_stats": selection_stats or {},
                "extraction_stats": {}
            }
        }

    async def auto_extract(self, text: str, available_schemas: List[SchemaMeta], max_retries: int = 3, method: str = "embedding") -> Dict[str, Any]:
        """Auto-detect schema and extract JSON from text."""
        start_time = time.time()
        logger.info(f"Starting auto-extraction process using {method} method")
        
        # Select best schema using specified method
        selection_result = await self.auto_detect_agent.select_schema(text, available_schemas, method)
        selected_schema_id = selection_result["schema_id"]
        selected_schema = selection_result["schema"]
        selection_stats = selection_result["usage_stats"]
        
        if not selected_schema_id or not selected_schema:
            logger.error("No valid schema found for auto-detection")
            return self._create_error_result(start_time, selection_stats)
        
        # Extract using selected schema
        extraction_start = time.time()
        result = await self.extractor.extract(text, selected_schema, max_retries)
        extraction_duration = time.time() - extraction_start
        
        # Enhance result with schema info and stats
        result.update({
            "schema_id": selected_schema_id,
            "schema_title": selected_schema.title,
            "schema_summary": selected_schema.summary,
            "stats": {
                "total_duration": time.time() - start_time,
                "selection_stats": selection_stats,
                "extraction_stats": {
                    "duration": extraction_duration,
                    "model": self.extractor.prompts['model_config']['extraction']['model_name']
                }
            }
        })
        
        logger.info(f"Auto-extraction completed: schema={selected_schema_id}, valid={result.get('valid')}")
        return result
    
    async def generate_schema_metadata(self, schema_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate metadata for a schema."""
        return await self.auto_detect_agent.generate_schema_metadata(schema_data)
    
    async def extract(self, text: str, schema: SchemaMeta, max_retries: int = 3) -> Dict[str, Any]:
        """Extract JSON from text using the configured extractor."""
        start_time = time.time()
        logger.info("ExtractorService.extract called")
        
        result = await self.extractor.extract(text, schema, max_retries)
        
        # Add performance metrics
        result["stats"] = {
            "duration": time.time() - start_time,
            "model": self.extractor.prompts['model_config']['extraction']['model_name']
        }
        
        logger.info(f"Extraction completed: valid={result.get('valid')}, attempts={result.get('attempts')}")
        return result
    
    def reset_conversation(self):
        """Reset the conversation history for a new extraction task."""
        logger.info("Resetting conversation through ExtractorService")
        if hasattr(self.extractor, 'reset_conversation'):
            self.extractor.reset_conversation()
    
    def get_conversation_history(self) -> List:
        """Get the current conversation history for debugging."""
        if hasattr(self.extractor, 'get_conversation_history'):
            return self.extractor.get_conversation_history()
        return []