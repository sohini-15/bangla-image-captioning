# Bangla Image Captioning — Backend

FastAPI backend running the full **ViT → GPT-2 → Bangla Translation** pipeline. Deployed on [HuggingFace Spaces](https://sohinim-bangla-image-captioning.hf.space).

## Architecture

```
Image Upload
    │
    ▼
┌─────────────────────────────────┐
│  ViT Encoder                    │
│  google/vit-base-patch16-224    │
│  768-dim visual embeddings      │
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
│  en → bn (deep-translator)     │
└───────────────┬─────────────────┘
                │
                ▼
        JSON Response or SSE Stream
```

## Model

**nlpconnect/vit-gpt2-image-captioning** from HuggingFace — a Vision Transformer encoder paired with a GPT-2 decoder, pretrained on image-caption pairs. Translation via `deep-translator` (Google Translate, en → bn). Evaluated against the BAN-Cap dataset using BLEU scores.

## API

### `POST /analyze-image`

Single-response captioning. Upload a file, get back JSON.

```json
{
  "filename": "photo.jpg",
  "caption": "a dog sitting on a bench in a park",
  "translation": "একটি কুকুর পার্কের বেঞ্চে বসে আছে",
  "insight": "The model identified this as a animal/wildlife image..."
}
```

### `POST /analyze-image-stream`

Same pipeline, but streams progress via Server-Sent Events. The frontend uses this endpoint to show real-time status as each stage completes.

Events: `status` (step progress), `result` (final output), `error` (if something fails).

### `GET /health`

Model loading status and device info.

## Run locally

```bash
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Model (~1GB) downloads on first run, then cached.

```bash
# Test it
python test_api.py path/to/image.jpg
```

## Deployment

This backend is deployed on **HuggingFace Spaces** (Docker SDK, CPU Basic, 16GB RAM). The Dockerfile pre-downloads the model at build time for fast cold starts. Previously deployed on Render, but the free tier's 512MB limit couldn't hold the model in memory.

## Stack

FastAPI · PyTorch · HuggingFace Transformers · deep-translator · Pillow · Docker