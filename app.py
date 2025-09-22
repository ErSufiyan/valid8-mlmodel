# filename: app.py
import base64
import json
import io
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image, ImageDraw
from ultralytics import YOLO  # ✅ Ultralytics YOLO

# ---------- CORS Setup ----------
origins = [
    "https://valdi8.netlify.app",  # your frontend
    "http://localhost:3000",       # optional, for local testing
]

# ---------- Load environment variables ----------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env")
genai.configure(api_key=GEMINI_API_KEY)

# ---------- Load YOLO Model ----------
MODEL_PATH = "best.pt"  # YOLO model
yolo_model = YOLO(MODEL_PATH)

# ---------- FastAPI App ----------
app = FastAPI(
    title="Gemini OCR + YOLO Model API",
    description="Validate, detect and extract certificate info from images",
    version="1.1"
)

# ---------- Add CORS Middleware ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Helper Functions ----------
def predict_and_draw(image_bytes):
    """Run YOLO on image and return (is_certificate, processed_image_bytes)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = yolo_model(img)

    # if YOLO detects nothing → not a certificate
    if len(results[0].boxes) == 0:
        return False, None

    # Draw bounding boxes
    draw = ImageDraw.Draw(img)
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.tolist())
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        draw.text((x1, y1 - 10), "Detected", fill="red")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return True, buf.read()

def extract_with_gemini(image_bytes: bytes):
    """Extract structured info from certificate using Gemini."""
    image_data = base64.b64encode(image_bytes).decode("utf-8")
    prompt = """
You are an expert OCR and information extraction system.
First, carefully check whether the given image is an educational marksheet or certificate 
(i.e., it should clearly contain information like name, roll number, course, branch, grades, 
or other educational details).

If it is NOT an educational certificate/marksheet, return the following JSON exactly:

{
  "error": "Please enter an educational certificate."
}

If it IS an educational certificate/marksheet, extract the following fields:
- Name
- Roll Number
- Course
- Branch
- Year
- CGPA
- SGPA
- Certificate Id
- Institution
- Issue Date

Return the result STRICTLY as a valid JSON object with keys exactly as above.
If a field is missing in the image, set its value to null.
Do not add extra commentary or explanation.
Only return JSON.
"""
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        contents=[{"role": "user", "parts": [prompt, {"mime_type": "image/png", "data": image_data}]}]
    )
    try:
        data = json.loads(response.text.strip())
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", response.text, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            data = {}
    return data

# ---------- API Endpoint ----------
@app.post("/extract/")
async def extract_certificate(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    image_bytes = await file.read()

    # Step 1: Run YOLO + draw
    is_certificate, processed_image = predict_and_draw(image_bytes)
    if not is_certificate:
        return JSONResponse(content={"error": "Please enter an educational certificate."})

    # Step 2: OCR extraction
    extracted_data = extract_with_gemini(image_bytes)

    # Step 3: Encode processed image to base64
    processed_image_b64 = base64.b64encode(processed_image).decode("utf-8") if processed_image else None

    return JSONResponse(content={
        "extracted_data": extracted_data,
        "processed_image": processed_image_b64
    })

# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
