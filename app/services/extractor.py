from typing import Protocol, Dict, Any, Optional
import json
import jsonschema
from abc import ABC, abstractmethod


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
    
    @abstractmethod
    async def _call_llm(self, text: str, schema: Dict[str, Any], attempt: int = 1, previous_error: Optional[str] = None) -> str:
        """Call the LLM and return raw JSON string."""
        pass

    @classmethod
    def _error_response(cls, error: str, max_retries: int) -> Dict[str, Any]:
        return {
            "data": {},
            "valid": False,
            "attempts": max_retries,
            "error": error
        }
    
    async def extract(self, text: str, schema: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """Extract JSON with validation and repair loop."""
        previous_error = None
        
        for attempt in range(1, max_retries + 1):
            try:
                # Call the LLM with previous error context
                raw_json = await self._call_llm(text, schema, attempt, previous_error)
                
                # Parse JSON
                try:
                    data = json.loads(raw_json)
                except json.JSONDecodeError as e:
                    error_msg = f"Invalid JSON: {str(e)}"
                    if attempt == max_retries:
                        return self._error_response(error_msg, max_retries)
                    previous_error = error_msg
                    continue
                
                # Validate against schema
                try:
                    jsonschema.validate(data, schema)
                    return {
                        "data": data,
                        "valid": True,
                        "attempts": attempt
                    }
                except jsonschema.ValidationError as e:
                    error_msg = f"Schema validation failed: {str(e)}"
                    if attempt == max_retries:
                        return self._error_response(error_msg, max_retries)
                    previous_error = error_msg
                    continue
                    
            except Exception as e:
                error_msg = f"Extraction failed: {str(e)}"
                if attempt == max_retries:
                    return self._error_response(error_msg, max_retries)
                previous_error = error_msg
                continue
        
        return self._error_response("Max retries exceeded", max_retries)

class GeminiExtractor(BaseExtractor):
    """Gemini-based JSON extractor."""
    
    def __init__(self, api_key: Optional[str] = None):
        from google import genai
        import os
        
        # Configure API key
        api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Google API key is required")
            
        self.client = genai.Client(api_key=api_key)
    
    async def _call_llm(self, text: str, schema: Dict[str, Any], attempt: int = 1, previous_error: Optional[str] = None) -> str:
        """Call Gemini to extract JSON."""
        schema_str = json.dumps(schema, indent=2)
        
        prompt = f"""You are a JSON extraction tool. Convert the following text to JSON that strictly follows the provided JSON schema.

JSON Schema:
{schema_str}

Text to extract from:
{text}

Return only valid JSON that matches the schema exactly. Do not include any explanation or additional text."""

        if attempt > 1 and previous_error:
            prompt += f"\n\nThis is attempt {attempt}. The previous attempt failed with this error: {previous_error}\n\nPlease fix this issue and ensure the JSON is valid and follows the schema precisely."
        elif attempt > 1:
            prompt += f"\n\nThis is attempt {attempt}. Please ensure the JSON is valid and follows the schema precisely."

        print(attempt)
        
        # Use synchronous call for now since the library may not support async properly
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        response = response.text.strip().replace("```json", "").replace("```", "")
        return response


class ExtractorService:
    """Service that manages JSON extraction using different LLM backends."""
    
    def __init__(self, extractor: Optional[ExtractorLLM] = None):
        self.extractor = extractor or GeminiExtractor()

    
    async def extract(self, text: str, schema: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """Extract JSON from text using the configured extractor."""
        return await self.extractor.extract(text, schema, max_retries)