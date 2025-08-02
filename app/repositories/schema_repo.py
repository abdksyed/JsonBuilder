from typing import Dict, List, Optional
import json
import os
from pathlib import Path
from app.models import SchemaMeta


class SchemaRepository:
    """Repository for managing JSON schemas with file-based storage."""
    
    def __init__(self, schemas_dir: str = "schemas"):
        self.schemas_dir = Path(schemas_dir)
        self.schemas_dir.mkdir(exist_ok=True)
        self._schemas: Dict[str, SchemaMeta] = {}
        self._load_schemas_from_files()
    
    def _load_schemas_from_files(self):
        """Load all schema files from the schemas directory."""
        self._schemas.clear()
        
        # Load all JSON files from schemas directory
        for schema_file in self.schemas_dir.glob("*.json"):
            try:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schema_data = json.load(f)
                
                # Create SchemaMeta from file data
                schema_meta = SchemaMeta(
                    id=schema_data.get("id", schema_file.stem),
                    title=schema_data.get("title", schema_file.stem),
                    summary=schema_data.get("summary", ""),
                    schema_data=schema_data.get("schema_data", {})
                )
                
                self._schemas[schema_meta.id] = schema_meta
                
            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                print(f"Warning: Failed to load schema from {schema_file}: {e}")
                continue
    
    def _save_schema_to_file(self, schema_meta: SchemaMeta, filename: str) -> None:
        """Save a schema to a JSON file."""
        # Ensure filename has .json extension
        if not filename.endswith('.json'):
            filename += '.json'
        
        # Sanitize filename to avoid path traversal
        filename = os.path.basename(filename)
        filepath = self.schemas_dir / filename
        
        schema_file_data = {
            "id": schema_meta.id,
            "title": schema_meta.title,
            "summary": schema_meta.summary,
            "schema_data": schema_meta.schema_data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(schema_file_data, f, indent=2, ensure_ascii=False)
    
    def _delete_schema_file(self, schema_id: str) -> bool:
        """Delete the schema file for a given schema ID."""
        # Find the file that contains this schema ID
        for schema_file in self.schemas_dir.glob("*.json"):
            try:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schema_data = json.load(f)
                
                if schema_data.get("id") == schema_id:
                    schema_file.unlink()  # Delete the file
                    return True
                    
            except (json.JSONDecodeError, FileNotFoundError):
                continue
        
        return False
    
    async def get_schema(self, schema_id: str) -> Optional[SchemaMeta]:
        """Get a schema by ID."""
        return self._schemas.get(schema_id)
    
    async def list_schemas(self, page: int = 1, per_page: int = 50) -> List[SchemaMeta]:
        """List schemas with pagination."""
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        schema_list = list(self._schemas.values())
        return schema_list[start_idx:end_idx]
    
    async def count_schemas(self) -> int:
        """Get total number of schemas."""
        return len(self._schemas)
    
    async def add_schema(self, schema_meta: SchemaMeta) -> None:
        """Add a new schema and save it to a file."""
        # Generate filename from title
        filename = '_'.join(schema_meta.title.split())
        
        # Save to file first
        self._save_schema_to_file(schema_meta, filename)
        
        # Add to in-memory cache
        self._schemas[schema_meta.id] = schema_meta
    
    async def delete_schema(self, schema_id: str) -> bool:
        """Delete a schema by ID."""
        if schema_id in self._schemas:
            # Remove from memory
            del self._schemas[schema_id]
            
            # Delete the file
            self._delete_schema_file(schema_id)
            
            return True
        return False
    
    async def reload_schemas(self) -> None:
        """Reload all schemas from files."""
        self._load_schemas_from_files()