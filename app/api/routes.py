from fastapi import APIRouter, HTTPException
from app.models import ExtractRequest, ExtractAutoRequest, ExtractResponse, SchemaListResponse, CreateSchemaRequest, CreateSchemaResponse, SchemaMeta
from app.services.extractor import ExtractorService
from app.repositories.schema_repo import SchemaRepository
from typing import Dict, Any

router = APIRouter()

# Initialize services (will be dependency injected later)
schema_repo = SchemaRepository()
extractor_service = ExtractorService()


@router.post("/extract", response_model=ExtractResponse)
async def extract_json(request: ExtractRequest) -> ExtractResponse:
    """Extract JSON from text using a specific schema."""
    try:
        # Get schema from repository
        schema = await schema_repo.get_schema(request.schema_id)
        if not schema:
            raise HTTPException(status_code=404, detail=f"Schema {request.schema_id} not found")
        
        # Extract JSON using the service
        result = await extractor_service.extract(request.text, schema)
        
        return ExtractResponse(
            schema_id=request.schema_id,
            data=result["data"],
            valid=result["valid"],
            attempts=result["attempts"],
            error=result.get("error"),
            stats=result.get("stats")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/select-schema")
async def select_schema(request: ExtractAutoRequest) -> Dict[str, Any]:
    """Auto-select best schema for given text (without extraction)."""
    try:
        # Get available schemas
        schemas = await schema_repo.list_schemas(page=1, per_page=request.top_k or 20)
        
        if not schemas:
            raise HTTPException(status_code=400, detail="No schemas available for auto-detection")
        
        # Auto-select schema only
        result = await extractor_service.auto_detect_agent.select_schema(request.text, schemas)
        
        return {
            "schema_id": result["schema_id"],
            "schema_title": result["schema"].title if result["schema"] else None,
            "schema_summary": result["schema"].summary if result["schema"] else None,
            "schema_data": result["schema"].schema_data if result["schema"] else None,
            "stats": result["usage_stats"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract/auto", response_model=ExtractResponse)
async def extract_json_auto(request: ExtractAutoRequest) -> ExtractResponse:
    """Auto-select best schema and extract JSON from text."""
    try:
        # Get available schemas
        schemas = await schema_repo.list_schemas(page=1, per_page=request.top_k or 20)
        
        if not schemas:
            raise HTTPException(status_code=400, detail="No schemas available for auto-detection")
        
        # Auto-detect and extract
        result = await extractor_service.auto_extract(request.text, schemas)
        
        return ExtractResponse(
            schema_id=result["schema_id"],
            data=result["data"],
            valid=result["valid"],
            attempts=result["attempts"],
            error=result.get("error"),
            stats=result.get("stats")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schemas/{schema_id}")
async def get_schema(schema_id: str) -> Dict[str, Any]:
    """Get a specific schema by ID."""
    try:
        schema = await schema_repo.get_schema(schema_id)
        if not schema:
            raise HTTPException(status_code=404, detail=f"Schema {schema_id} not found")
        return schema.schema_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schemas", response_model=SchemaListResponse)
async def list_schemas(page: int = 1, per_page: int = 50) -> SchemaListResponse:
    """List all available schemas with pagination."""
    try:
        schemas = await schema_repo.list_schemas(page=page, per_page=per_page)
        total = await schema_repo.count_schemas()
        
        return SchemaListResponse(
            schemas=schemas,
            total=total,
            page=page,
            per_page=per_page
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schemas", response_model=CreateSchemaResponse)
async def create_schema(request: CreateSchemaRequest) -> CreateSchemaResponse:
    """Create a new schema."""
    try:
        import jsonschema
        
        # Validate that the schema_data is a valid JSON schema
        try:
            jsonschema.Draft7Validator.check_schema(request.schema_data)
        except jsonschema.SchemaError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON schema: {str(e)}")
        
        # Generate metadata if not provided
        title = request.title
        summary = request.summary
        schema_id = None
        
        if not title or not summary:
            metadata = await extractor_service.generate_schema_metadata(request.schema_data)
            if not title:
                title = metadata["title"]
            if not summary:
                summary = metadata["summary"]
            schema_id = metadata["slug"]
        
        # If we still don't have an ID, generate one from title
        if not schema_id:
            import re
            schema_id = re.sub(r'[^a-z0-9-]', '-', title.lower())
            schema_id = re.sub(r'-+', '-', schema_id).strip('-')
        
        # Create schema metadata
        schema_meta = SchemaMeta(
            id=schema_id,
            title=title,
            summary=summary,
            schema_data=request.schema_data
        )
        
        # Save to repository
        await schema_repo.add_schema(schema_meta)
        
        return CreateSchemaResponse(
            id=schema_id,
            message="Schema created successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/schemas/{schema_id}")
async def delete_schema(schema_id: str) -> Dict[str, str]:
    """Delete a schema by ID."""
    try:
        success = await schema_repo.delete_schema(schema_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Schema {schema_id} not found")
        
        return {"message": "Schema deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))