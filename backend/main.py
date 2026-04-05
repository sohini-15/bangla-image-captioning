from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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
        "caption": "A person standing outdoors near a building.",
        "translation": "Ekti manush ekti bhoboner kache baire dariye ache.",
        "insight": "The person is standing in a natural setting with trees and sky visible."
    }