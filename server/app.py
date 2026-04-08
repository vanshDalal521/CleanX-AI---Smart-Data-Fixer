import os
try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required for the web interface.") from e

try:
    from ..models import CleanxAction, CleanxObservation
    from .cleanx_environment import CleanxEnvironment
except (ModuleNotFoundError, ImportError):
    from models import CleanxAction, CleanxObservation
    from server.cleanx_environment import CleanxEnvironment

app = create_app(
    CleanxEnvironment,
    CleanxAction,
    CleanxObservation,
    env_name="cleanx",
    max_concurrent_envs=10,
)

# Custom CleanX AI UI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

ui_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui")
os.makedirs(ui_dir, exist_ok=True)

app.mount("/ui", StaticFiles(directory=ui_dir, html=True), name="ui")

@app.get("/")
async def redirect_to_ui():
    return FileResponse(os.path.join(ui_dir, "index.html"))

from fastapi import UploadFile, File, Form
from server import cleanx_environment
import pandas as pd
import io
import subprocess
import os

@app.post("/upload_custom")
async def upload_custom(file: UploadFile = File(...), goal: str = Form("Clean this dataset.")):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        cleanx_environment.LATEST_CUSTOM_DATA = df
        cleanx_environment.LATEST_CUSTOM_GOAL = goal
        return {"status": "success"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/run_ai_demo")
async def run_ai_demo(api_key: str = Form(...)):
    env_vars = os.environ.copy()
    env_vars["OPENAI_API_KEY"] = api_key
    try:
        proc = subprocess.run(
            ["python", "inference.py"], 
            capture_output=True, text=True, env=env_vars, timeout=60, cwd=os.path.dirname(ui_dir)
        )
        return {"output": proc.stdout, "error": proc.stderr}
    except Exception as e:
        return {"error": str(e)}

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port) # main()

