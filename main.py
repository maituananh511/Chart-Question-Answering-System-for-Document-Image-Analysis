

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import io
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

from config import settings
from pipeline import ChartQAPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

pipeline: ChartQAPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Loading models, please wait...")
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    pipeline = ChartQAPipeline()
    logger.info("All models loaded. API ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Chart QA API",
    description="Chart Question Answering with multi-turn chat",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



def _read_image(contents: bytes, content_type: str) -> Image.Image:
    if content_type == "application/pdf":
        try:
            import fitz
        except ImportError:
            raise HTTPException(status_code=500, detail="Cần cài pymupdf: pip install pymupdf")
        doc = fitz.open(stream=io.BytesIO(contents), filetype="pdf")
        if len(doc) == 0:
            raise HTTPException(status_code=400, detail="PDF không có trang nào.")
        pix = doc[0].get_pixmap(dpi=150)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return Image.open(io.BytesIO(contents)).convert("RGB")


def _validate_upload(image: UploadFile, contents: bytes):
    allowed = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    if image.content_type not in allowed:
        raise HTTPException(status_code=400,
            detail=f"Chỉ chấp nhận JPEG/PNG/WEBP/JPG, nhận được: {image.content_type}")
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.MAX_IMAGE_SIZE_MB:
        raise HTTPException(status_code=400,
            detail=f"File quá lớn ({size_mb:.1f}MB), tối đa {settings.MAX_IMAGE_SIZE_MB}MB")



@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": settings.DEVICE,
        "sessions": len(pipeline.list_sessions()) if pipeline else 0,
    }


@app.post("/api/upload")
async def upload(
    image: UploadFile = File(..., description="Ảnh biểu đồ (PNG, JPG, JPEG, PDF)"),
):
    """
    Upload ảnh biểu đồ → classify + extract → trả về session_id.
    Dùng session_id này để chat nhiều lượt mà không cần upload lại.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline chưa sẵn sàng.")

    contents = await image.read()
    _validate_upload(image, contents)

    try:
        pil_image = _read_image(contents, image.content_type)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không thể đọc file: {e}")

    result = pipeline.upload_image(pil_image)

    return JSONResponse(content={
        "session_id"    : result.session_id,
        "chart_type"    : result.chart_type,
        "extracted_data": result.extracted_data,
        "supported"     : result.supported,
        "latency"       : result.latency,
        "message"       : result.answer if not result.supported else
                          f"Ảnh đã được phân tích. Loại biểu đồ: {result.chart_type}. Hãy đặt câu hỏi!",
    })


@app.post("/api/chat")
async def chat(
    session_id: str = Form(..., description="Session ID từ /api/upload"),
    question  : str = Form(..., description="Câu hỏi về biểu đồ"),
):
    """
    Chat với biểu đồ đã upload. Vintern nhớ history các lượt trước.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline chưa sẵn sàng.")

    if not question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")

    result = pipeline.chat(session_id=session_id, question=question)

    return JSONResponse(content={
        "session_id"    : session_id,
        "question"      : question,
        "answer"        : result.answer,
        "chart_type"    : result.chart_type,
        "supported"     : result.supported,
        "latency"       : result.latency,
    })


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline chưa sẵn sàng.")
    pipeline.delete_session(session_id)
    return {"message": f"Session {session_id} đã được xóa."}


@app.post("/api/ask")
async def ask(
    image   : UploadFile = File(...),
    question: str = Form(...),
):
    """Single-turn (backward compat với UI cũ)."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline chưa sẵn sàng.")

    contents = await image.read()
    _validate_upload(image, contents)

    try:
        pil_image = _read_image(contents, image.content_type)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không thể đọc file: {e}")

    result = pipeline.run(image=pil_image, question=question)

    return JSONResponse(content={
        "question"      : question,
        "answer"        : result.answer,
        "chart_type"    : result.chart_type,
        "extracted_data": result.extracted_data,
        "latency"       : result.latency,
        "supported"     : result.supported,
    })


@app.get("/", response_class=HTMLResponse)
async def ui():
    for html_path in [
        Path(__file__).parent / "static" / "index.html",
        Path(__file__).parent / "index.html",
    ]:
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Chart QA API v2</h1><p><a href='/docs'>/docs</a></p>")
