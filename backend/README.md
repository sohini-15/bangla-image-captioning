# Bangla Image Captioning — Backend

A FastAPI backend that runs a complete **ViT → GPT-2 → Bangla Translation** pipeline for image captioning.

## Architecture

```
Image Upload
    │
    ▼
┌─────────────────────────────────┐
│  ViT Encoder                    │
│  (google/vit-base-patch16-224)  │
│  Extracts 768-dim visual        │
│  feature embeddings             │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  GPT-2 Decoder                  │
│  Cross-attention on visual      │
│  tokens → English caption       │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Google Translate API           │
│  English → Bangla (bn)          │
│  তারপর বাংলায় অনুবাদ           │
└───────────────┬─────────────────┘
                │
                ▼
        JSON Response
   { caption, translation,
     filename, insight }
```

## Model

**nlpconnect/vit-gpt2-image-captioning** from HuggingFace

- **Encoder**: Vision Transformer (ViT) — splits the image into 16×16 patches and encodes them into a sequence of 768-dimensional embeddings
- **Decoder**: GPT-2 — uses cross-attention over the ViT embeddings to autoregressively generate an English caption
- **Translation**: googletrans (Google Translate API wrapper) converts the English caption to Bangla

Evaluated against the **BAN-Cap** dataset (Bangla captions for Flickr8k) using BLEU scores.

## API

### `POST /analyze-image`

Upload an image file and receive captions.

**Request**: `multipart/form-data` with a `file` field containing an image (JPG, PNG, WEBP)

**Response**:
```json
{
  "filename": "photo.jpg",
  "caption": "a dog sitting on a bench in a park",
  "translation": "একটি কুকুর পার্কের বেঞ্চে বসে আছে",
  "insight": "The model identified this as a animal/wildlife image..."
}
```

### `GET /health`

Returns model loading status and device info.

### `GET /`

API info and pipeline description.

## Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The model downloads automatically on first startup (~1GB). Subsequent starts use the cached model.

**Test it:**
```bash
curl -X POST http://localhost:8000/analyze-image \
  -F "file=@your_image.jpg"
```

## Deploy to Render

1. Push this `backend/` folder to your GitHub repo
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your repo, set the root directory to `backend/`
4. Environment: **Docker**
5. Render auto-detects the `Dockerfile` and builds

The Dockerfile pre-downloads the model at build time so cold starts are fast.

**For GPU inference** (faster, but costs more): select a GPU instance on Render. The code auto-detects CUDA availability.

**For CPU inference** (free tier): works fine, inference takes ~2-5 seconds per image.

## Stack

- **FastAPI** — async web framework
- **PyTorch** — model inference
- **HuggingFace Transformers** — ViT-GPT2 model loading and generation
- **googletrans** — Google Translate API for Bangla translation
- **Pillow** — image processing
- **Docker** — containerized deployment

## Frontend

The companion frontend is a Next.js app deployed on Vercel. It sends a `POST /analyze-image` request with the uploaded file and displays the results.

Frontend repo: `frontend/`