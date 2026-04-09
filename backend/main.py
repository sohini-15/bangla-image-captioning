from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://bangla-image-captioning.vercel.app",
        "http://localhost:3000"
    ],
    allow_credentials = True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Backend ok"}

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    return {
        "filename": file.filename,
        "caption": "A sample caption for the uploaded image.",
        "translation": "একটি নমুনা বর্ণনা",
        "insight": "This is a demo response for testing purposes."
    }