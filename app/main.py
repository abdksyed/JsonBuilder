from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.routes import router
import os

app = FastAPI(
    title="Schema-Aware JSON Extractor",
    description="Extract structured JSON from text using JSON Schema validation",
    version="0.9.0"
)

app.include_router(router)

# Mount static files
static_path = os.path.join(os.path.dirname(__file__), "ui", "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# Serve the UI
@app.get("/")
async def serve_ui():
    template_path = os.path.join(os.path.dirname(__file__), "ui", "templates", "index.html")
    return FileResponse(template_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)