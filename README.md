# বাংলা Image Captioning

**Upload any image. Get a caption in English and বাংলা. Powered by a real ML pipeline.**

<br>

### 🔗 [Try the live demo →](https://bangla-image-captioning.vercel.app)

<br>

---

<br>

## What it does

Drop an image into the app. A Vision Transformer encodes it, GPT-2 generates an English caption from the visual features, and Google Translate converts that caption to Bangla. The whole pipeline runs end-to-end on a single API call — no hardcoded responses, no shortcuts.

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
| **Backend** | FastAPI, PyTorch, HuggingFace Transformers |
| **Frontend** | Next.js, React, TypeScript, Tailwind CSS |
| **Deployment** | Frontend on Vercel, Backend on Render (Docker) |
| **Evaluation** | BLEU score against the BAN-Cap dataset (Bangla captions for Flickr8k) |

<br>

## Project structure

```
bangla-image-captioning/
│
├── backend/
│   ├── main.py              ← FastAPI app running the full ViT → GPT-2 → Translate pipeline
│   ├── requirements.txt
│   ├── Dockerfile           ← Production container, pre-downloads the model at build time
│   ├── test_api.py          ← Automated test script
│   └── README.md
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx         ← Main UI — image upload, result display
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── package.json
│   └── ...
│
└── README.md                ← You are here
```

<br>

## Why this exists

Most image captioning systems only work in English. For the ~230 million people who speak Bangla, that's a gap. This project bridges it by combining a state-of-the-art vision-language model with machine translation — producing Bangla captions that are evaluated against human-written references from the BAN-Cap dataset using BLEU scores.

It's also a demonstration of building an end-to-end ML system: from model selection and inference, through translation, to a deployed web application that anyone can use.

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

## Evaluation

Captions are evaluated against the **BAN-Cap** dataset using corpus-level BLEU scores, comparing the machine-translated Bangla output against human-written Bangla reference captions for Flickr8k images. The evaluation notebook with full methodology is available in the repository.

<br>

---

Built by **MJ** · MS Computer Engineering, Clemson University · BTech ECE, Heritage Institute of Technology
