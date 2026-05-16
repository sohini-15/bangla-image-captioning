"""
Bangla Image Captioning — Backend
==================================
FastAPI backend that runs the full ML pipeline:
  1. ViT Encoder (google/vit-base-patch16-224) encodes the uploaded image
  2. GPT-2 Decoder generates an English caption via cross-attention
  3. Google Translate API translates the caption to Bangla (bn)

Model: nlpconnect/vit-gpt2-image-captioning (HuggingFace)
Dataset used for evaluation: BAN-Cap (Bangla captions for Flickr8k)

Author: MJ
Stack: FastAPI · PyTorch · HuggingFace Transformers · googletrans
"""

import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from transformers import ViTImageProcessor, GPT2TokenizerFast, VisionEncoderDecoderModel
from deep_translator import GoogleTranslator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model references (loaded once at startup)
# ---------------------------------------------------------------------------
model = None
image_processor = None
tokenizer = None
translator = None
device = None

MODEL_ID = "nlpconnect/vit-gpt2-image-captioning"


def load_models():
    """Load the ViT-GPT2 model, processor, tokenizer, and translator."""
    global model, image_processor, tokenizer, translator, device

    logger.info("Loading ViT-GPT2 image captioning model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    image_processor = ViTImageProcessor.from_pretrained(MODEL_ID)
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()

    translator = GoogleTranslator(source="en", target="bn")

    logger.info("All models loaded successfully.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    load_models()
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Bangla Image Captioning API",
    description="ViT-GPT2 image captioning with Bangla translation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://bangla-image-captioning.vercel.app",
        "http://localhost:3000",
        "http://localhost:3001",
        "*",  # During development — tighten for production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pipeline functions (from the original notebook)
# ---------------------------------------------------------------------------

def generate_caption(image: Image.Image, greedy: bool = True) -> str:
    """
    Generate an English caption for the given image using ViT-GPT2.

    This replicates the `show_n_generate` function from the notebook:
      - ViT encodes the image into pixel values
      - GPT-2 decodes an English caption via cross-attention on visual tokens
    """
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Process image through ViT
    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Generate caption with GPT-2 decoder
    with torch.no_grad():
        if greedy:
            generated_ids = model.generate(pixel_values, max_new_tokens=30)
        else:
            generated_ids = model.generate(
                pixel_values,
                do_sample=True,
                max_new_tokens=30,
                top_k=5,
            )

    caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption.strip()


def translate_to_bangla(text: str) -> str:
    """
    Translate English text to Bangla using Google Translate API.

    Replicates the `translate_to_bangla` function from the notebook.
    """
    try:
        result = translator.translate(text)
        return result
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return f"[Translation unavailable] {text}"


def generate_insight(caption: str) -> str:
    """
    Generate a short machine-generated insight about the image
    based on the caption. This adds context about what the pipeline
    detected and how it interpreted the visual content.
    """
    # Simple rule-based insight generation
    words = caption.lower().split()
    num_tokens = len(words)

    # Detect broad categories from caption content
    animal_words = {"dog", "cat", "bird", "horse", "cow", "sheep", "elephant",
                    "bear", "zebra", "giraffe", "fish", "animal", "puppy", "kitten"}
    people_words = {"man", "woman", "boy", "girl", "person", "people", "child",
                    "player", "team", "group", "couple", "baby"}
    outdoor_words = {"street", "road", "beach", "mountain", "field", "park",
                     "ocean", "river", "lake", "sky", "tree", "forest", "garden"}
    food_words = {"food", "pizza", "cake", "plate", "table", "kitchen",
                  "restaurant", "eating", "meal", "dinner", "lunch", "breakfast"}
    vehicle_words = {"car", "bus", "truck", "train", "motorcycle", "bicycle",
                     "boat", "plane", "airplane"}

    detected = set(words)

    if detected & animal_words:
        category = "animal/wildlife"
    elif detected & people_words:
        category = "people/activity"
    elif detected & vehicle_words:
        category = "vehicle/transportation"
    elif detected & food_words:
        category = "food/dining"
    elif detected & outdoor_words:
        category = "outdoor/landscape"
    else:
        category = "general scene"

    return (
        f"The model identified this as a {category} image. "
        f"The ViT encoder extracted visual features which the GPT-2 decoder "
        f"interpreted into a {num_tokens}-token English description, "
        f"then translated to Bangla via Google Translate API."
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Bangla Image Captioning API is running",
        "model": MODEL_ID,
        "pipeline": "ViT Encoder → GPT-2 Decoder → Google Translate (en→bn)",
        "device": str(device) if device else "not loaded",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown",
    }


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Full captioning pipeline:
      1. Read uploaded image
      2. Generate English caption (ViT-GPT2)
      3. Translate to Bangla (Google Translate)
      4. Generate insight
      5. Return results matching the frontend's expected schema
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image.",
        )

    try:
        # 1. Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        logger.info(f"Received image: {file.filename} ({image.size}, {image.mode})")

        # 2. Generate English caption using ViT-GPT2
        #    Using greedy decoding for consistent results (matching notebook default)
        caption_en = generate_caption(image, greedy=True)
        logger.info(f"English caption: {caption_en}")

        # 3. Translate to Bangla
        caption_bn = translate_to_bangla(caption_en)
        logger.info(f"Bangla translation: {caption_bn}")

        # 4. Generate insight
        insight = generate_insight(caption_en)

        # 5. Return in the shape the frontend expects
        return {
            "filename": file.filename,
            "caption": caption_en,
            "translation": caption_bn,
            "insight": insight,
        }

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


# ---------------------------------------------------------------------------
# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# ---------------------------------------------------------------------------