from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class ExtractRequest(BaseModel):
    text: str
    schema_id: str


class ExtractAutoRequest(BaseModel):
    text: str
    top_k: Optional[int] = 20
    method: Optional[str] = "embedding"  # "embedding" or "llm"


class ExtractResponse(BaseModel):
    schema_id: str
    data: Dict[str, Any]
    valid: bool
    attempts: int
    error: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None


class SchemaMeta(BaseModel):
    id: str
    title: str
    summary: str
    schema_data: Dict[str, Any]


class SchemaListResponse(BaseModel):
    schemas: List[SchemaMeta]
    total: int
    page: int
    per_page: int


class CreateSchemaRequest(BaseModel):
    title: Optional[str] = None
    summary: Optional[str] = None
    schema_data: Dict[str, Any]


class CreateSchemaResponse(BaseModel):
    id: str
    message: str