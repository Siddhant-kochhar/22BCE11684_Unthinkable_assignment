"""
Simple test server to verify FastAPI is working
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI(title="Test Server")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Test server is working!", "status": "ok"}

@app.get("/test", response_class=HTMLResponse)
async def test_html():
    return """
    <html>
        <head>
            <title>Test Page</title>
        </head>
        <body>
            <h1>FastAPI Test Server is Working!</h1>
            <p>This means the server is running correctly.</p>
            <a href="/docs">API Documentation</a>
        </body>
    </html>
    """

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "Server is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("test_server:app", host="0.0.0.0", port=8001, reload=True)