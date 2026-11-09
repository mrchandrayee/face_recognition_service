from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil, uuid
from .config import CONFIG, UPLOAD_DIR, OUTPUT_DIR, GALLERY_DIR
from .recognition import FaceService
from .processor import process_video

app = FastAPI(title=CONFIG["app"]["title"])
app.mount("/static", StaticFiles(directory=Path(__file__).parent/"static"), name="static")
face_service = None

@app.on_event("startup")
def startup():
    global face_service
    face_service = FaceService()

@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).parent/"static"/"index.html").read_text(encoding="utf-8")

@app.get("/api/status")
def status():
    return {"service":"ready","enrolled_people":sorted(face_service.gallery.keys()) if face_service else []}

@app.post("/api/enroll")
async def enroll(name: str = Form(...), photo: UploadFile = File(...)):
    data = await photo.read()
    try:
        saved = face_service.enroll(name, data)
        return {"ok":True,"name":saved,"enrolled_people":sorted(face_service.gallery)}
    except ValueError as exc:
        raise HTTPException(400,str(exc))

@app.post("/api/process")
async def process(video: UploadFile = File(...)):
    max_bytes = CONFIG["app"]["max_upload_mb"] * 1024 * 1024
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    size = 0
    with path.open("wb") as out:
        while chunk := await video.read(1024*1024):
            size += len(chunk)
            if size > max_bytes:
                path.unlink(missing_ok=True)
                raise HTTPException(413,"Video exceeds upload size limit.")
            out.write(chunk)
    try:
        report = process_video(str(path), face_service, f"{uuid.uuid4().hex}.mp4")
        report["download_url"] = "/api/output/" + Path(report["output"]).name
        return report
    except Exception as exc:
        raise HTTPException(400, f"Video processing failed: {exc}")
    finally:
        path.unlink(missing_ok=True)

@app.get("/api/output/{filename}")
def output(filename: str):
    path = OUTPUT_DIR / Path(filename).name
    if not path.exists(): raise HTTPException(404,"Output not found")
    return FileResponse(path, media_type="video/mp4", filename=path.name)
