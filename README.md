# Chart QA Pipeline

Hệ thống hỏi đáp thông minh về biểu đồ sử dụng 3 model AI:

```
Ảnh biểu đồ + Câu hỏi
        ↓
   ResNet18(Classify chart)
        ↓
   PaddleOCR-VL (extract data từ chart)
        ↓
   Vintern-1B (trả lời câu hỏi)
        ↓
      Kết quả
```

---

## Cấu trúc thư mục

```
your/path/to/files
├── paddle_server.py          # Server riêng cho PaddleOCR-VL (port 8001)
├── venv_paddle\              # Virtual env riêng cho Paddle (transformers>=4.45)
│
└── files\
    ├── main.py               # FastAPI app chính (port 8000)
    ├── pipeline.py           # Orchestrator điều phối 3 bước
    ├── config.py             # Cấu hình paths, device, params
    ├── chart_classifier.py   #  classifier
    ├── data_extractor.py     # HTTP client gọi paddle_server:8001
    ├── chart_qa.py           # Vintern QA engine
    ├── index.html            # Web UI
    ├── requirements.txt
    ├── models_local\
    │   ├── vintern\          # Vintern-1B-v2 weights
    │   └── paddleocr_vl\     # PaddleOCR-VL weights
        └──resnet18
    └── venv\                 # Virtual env chính (transformers==4.44.2)
```

---

## Lý do dùng 2 virtual environment

PaddleOCR-VL và Vintern yêu cầu **2 version transformers xung đột nhau**:

| | venv (Main) | venv_paddle (Paddle) |
|---|---|---|
| transformers | `==4.44.2` | `>=4.45` |
| Chứa | Vintern + FastAPI chính | PaddleOCR-VL + FastAPI mini |
| Port | 8000 | 8001 |

---

## Cài đặt lần đầu

### 1. Tải model về local

```powershell
cd your/path/to/files

# Vintern
python -c "from huggingface_hub import snapshot_download; snapshot_download('5CD-AI/Vintern-1B-v3_5', local_dir='./models_local/vintern')"

# PaddleOCR-VL
python -c "from huggingface_hub import snapshot_download; snapshot_download('PaddlePaddle/PaddleOCR-VL', local_dir='./models_local/paddleocr_vl')"
```

### 2. Tạo venv_paddle

```powershell
cd your/path/to/files

python -m venv venv_paddle
venv_paddle\Scripts\activate

python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install "transformers>=4.45" accelerate pillow fastapi uvicorn python-multipart httpx
```

### 3. Cài dependencies cho venv chính

```powershell
cd your/path/to/files
venv\Scripts\activate

python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install "transformers==4.44.2" accelerate pillow fastapi uvicorn python-multipart httpx ultralytics timm pydantic-settings
```

---

## Chạy hệ thống

Mỗi lần chạy cần **2 terminal**:

### Terminal 1 — Paddle Server

```powershell
cd your/path/to/files
venv_paddle\Scripts\activate
python -m uvicorn paddle_server:app --port 8001
```

Chờ thấy:
```
PaddleOCR-VL ready on cuda
```

### Terminal 2 — Main App

```powershell
cd your/path/to/files
venv\Scripts\activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Chờ thấy:
```
Pipeline ready in ~39s
All models loaded. API ready.
```

### Mở Web UI

Truy cập: **http://localhost:8000**

---

## Sử dụng

1. Mở **http://localhost:8000**
2. Upload ảnh biểu đồ (PNG, JPG, WEBP — tối đa 10MB)
3. Nhập câu hỏi (tiếng Việt hoặc tiếng Anh)
4. Nhấn **Phân tích & Trả lời**
5. Xem kết quả gồm: chart type, extracted data, câu trả lời, latency

---

## API Endpoint

### `POST /api/ask`

```bash
curl -X POST http://localhost:8000/api/ask \
  -F "image=@chart.png" \
  -F "question=Công ty nào có doanh thu cao nhất?"
```

Response:
```json
{
  "question": "Công ty nào có doanh thu cao nhất?",
  "answer": "FPT có doanh thu cao nhất.",
  "chart_type": "h_bar",
  "extracted_data": "...",
  "latency": {
    "classify": 6.24,
    "extract": 45.2,
    "qa": 30.1,
    "total": 81.5
  }
}
```

### `GET /health`

Kiểm tra trạng thái models.

### `GET /docs`

Swagger UI để test API trực tiếp trên browser.

---

## Cấu hình

Chỉnh trong `config.py`:

```python
Resnet_MODEL_PATH   = "Resnet18/best.pt"
PADDLE_MODEL_PATH = "./models_local/paddleocr_vl"
VINTERN_MODEL_PATH = "./models_local/vintern"
DEVICE = "cuda"
VINTERN_MAX_NEW_TOKENS = 1024
PADDLE_MAX_NEW_TOKENS  = 512
```

---

## Lưu ý

- **GPU cần thiết**: Cả 2 model đều chạy trên CUDA, không có GPU sẽ rất chậm
- **VRAM**: Cần khoảng 8-12GB VRAM tổng cho cả Paddle + Vintern
- **Thời gian xử lý**: Mỗi request mất khoảng 1-3 phút tùy độ phức tạp của biểu đồ
- **Paddle timeout**: Nếu bị timeout, tăng giá trị `timeout=300.0` trong `data_extractor.py`
- **Web UI**: File `index.html` phải nằm cùng cấp với `main.py` hoặc trong thư mục `static/`