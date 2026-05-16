# Bangla Image Captioning — Frontend

Next.js frontend for the Bangla Image Captioning pipeline. Deployed on [Vercel](https://bangla-image-captioning.vercel.app).

## What it does

Users upload an image and watch the ML pipeline process it in real time. The frontend streams Server-Sent Events from the backend, displaying live progress as the image passes through each stage: ViT encoding → GPT-2 caption generation → Bangla translation. Results appear as they're generated — English caption, বাংলা translation, and a model insight.

## Features

- Image upload with preview
- Live pipeline status bar showing each processing stage
- Real-time log console streaming backend progress via SSE
- Animated progress indicator
- Responsive layout

## Stack

Next.js · React · TypeScript · Tailwind CSS · Vercel

## Run locally

```bash
npm install
npm run dev
```

Opens at [http://localhost:3000](http://localhost:3000). By default points to the production backend on HuggingFace Spaces. To use a local backend, change `API_BASE` in `app/page.tsx` to `http://localhost:8000`.

## Connects to

Backend API on HuggingFace Spaces: [sohinim-bangla-image-captioning.hf.space](https://sohinim-bangla-image-captioning.hf.space)