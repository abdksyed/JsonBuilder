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
        result = await extractor_service.extract(request.text, schema.schema_data)
        
        return ExtractResponse(
            schema_id=request.schema_id,
            data=result["data"],
            valid=result["valid"],
            attempts=result["attempts"],
            error=result.get("error")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract/auto", response_model=ExtractResponse)
async def extract_json_auto(request: ExtractAutoRequest) -> ExtractResponse:
    """Auto-select best schema and extract JSON from text."""
    try:
        # TODO: Implement schema selection logic
        # For now, just return an error
        raise HTTPException(status_code=501, detail="Auto-extraction not yet implemented")
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
        import uuid
        import jsonschema
        
        # Validate that the schema_data is a valid JSON schema
        try:
            jsonschema.Draft7Validator.check_schema(request.schema_data)
        except jsonschema.SchemaError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON schema: {str(e)}")
        
        # Generate unique ID
        schema_id = str(uuid.uuid4())
        
        # Create schema metadata
        schema_meta = SchemaMeta(
            id=schema_id,
            title=request.title,
            summary=request.summary,
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