# বাংলা Image Captioning

**Upload any image. Get a caption in English and বাংলা. Powered by a real ML pipeline — no hardcoded responses, no shortcuts.**

<br>

### 🔗 [Try the live demo →](https://bangla-image-captioning.vercel.app)

<br>

---

<br>

## What it does

Drop an image into the app. A Vision Transformer encodes it into visual features, GPT-2 generates an English caption through cross-attention on those features, and Google Translate converts the caption to Bangla. The frontend streams real-time progress from the backend via Server-Sent Events, so you can watch each pipeline stage as it runs.

<br>

## How the pipeline works

```
                    ┌──────────────────────────────┐
  Image (RGB) ───▶  │  ViT Encoder                 │
                    │  google/vit-base-patch16-224  │
                    │  224×224 → 16×16 patches      │
                    │  → 768-dim embeddings         │
                    └─────────────┬────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────────┐
                    │  GPT-2 Decoder               │
                    │  Cross-attention on ViT       │
                    │  embeddings → autoregressive  │
                    │  English caption generation   │
                    └─────────────┬────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────────┐
                    │  Google Translate API         │
                    │  en → bn                     │
                    │  English → বাংলা              │
                    └─────────────┬────────────────┘
                                  │
                                  ▼
                          JSON response
                   { caption, translation, insight }
```

<br>

## The stack

| Layer | Tech |
|-------|------|
| **Model** | `nlpconnect/vit-gpt2-image-captioning` — ViT encoder + GPT-2 decoder, pretrained on image-caption pairs |
| **Translation** | Google Translate API via `deep-translator` (en → bn) |
| **Backend** | FastAPI, PyTorch, HuggingFace Transformers, Server-Sent Events for real-time streaming |
| **Frontend** | Next.js, React, TypeScript, Tailwind CSS |
| **Deployment** | Frontend on Vercel · Backend on HuggingFace Spaces (Docker, 16GB RAM) |
| **Evaluation** | BLEU score against the BAN-Cap dataset (Bangla captions for Flickr8k) |

<br>

## Project structure

```
bangla-image-captioning/
│
├── backend/                     ← FastAPI app (also deployed separately to HF Spaces)
│   ├── main.py                  ← Full ViT → GPT-2 → Translate pipeline with SSE streaming
│   ├── requirements.txt
│   ├── Dockerfile               ← Production container, pre-downloads the ~1GB model at build time
│   └── test_api.py              ← Automated test script
│
├── frontend/                    ← Next.js app deployed on Vercel
│   ├── app/
│   │   ├── page.tsx             ← Image upload, live pipeline status, result display
│   │   ├── layout.tsx
│   │   └── globals.css
│   └── package.json
│
├── hf-bangla/                   ← HuggingFace Spaces deployment (mirrors backend/)
│
└── README.md                    ← You are here
```

<br>

## Why this exists

Most image captioning systems only work in English. For the ~230 million people who speak Bangla, that's a gap. This project bridges it by combining a state-of-the-art vision-language model with machine translation, producing Bangla captions that are evaluated against human-written references from the BAN-Cap dataset using BLEU scores.

It's also a demonstration of building an end-to-end ML system: from model selection and inference, through translation, to a deployed web application with real-time progress streaming that anyone can use.

<br>

## Deployment architecture

The backend was initially deployed on Render, but the free tier's 512MB RAM limit couldn't hold the ~1GB ViT-GPT2 model in memory. The backend was moved to HuggingFace Spaces (Docker SDK, CPU Basic tier with 16GB RAM), which runs the full model comfortably and keeps the project within the ML ecosystem. The frontend remains on Vercel and communicates with the HF Spaces backend over HTTPS.

<br>

## Run it locally

```bash
# Backend
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --port 8000

# Frontend (in another terminal)
cd frontend
npm install
npm run dev
```

The model (~1GB) downloads automatically on first backend startup. After that, it's cached.

<br>

## API

The backend exposes two captioning endpoints:

`POST /analyze-image` — returns the full result in a single JSON response.

`POST /analyze-image-stream` — streams real-time progress via Server-Sent Events as each pipeline stage completes (image preprocessing → ViT encoding → GPT-2 decoding → Bangla translation), then sends the final result. The frontend uses this endpoint.

`GET /health` — returns model loading status and device info.

<br>

---

Built by **Sohini** · MS Computer Engineering, Clemson University · BTech ECE, Heritage Institute of Technology
Linkedin: https://www.linkedin.com/in/sohinimazumder15