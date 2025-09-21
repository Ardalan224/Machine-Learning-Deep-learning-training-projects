from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
from PIL import Image, ImageOps

# Load pretrained model
model = joblib.load("digits_KNN_model.pkl")

app = FastAPI()

@app.get("/")
def main():
    return HTMLResponse(content="""
    <html>
        <head>
            <title>Digit Classifier</title>
        </head>
        <body>
            <h2>Upload a handwritten digit image (PNG/JPG)</h2>
            <form id="upload-form">
                <input id="file-input" type="file" name="file" accept="image/*">
                <input type="submit" value="Predict">
            </form>
            <h3 id="result"></h3>

            <script>
                const form = document.getElementById('upload-form');
                form.addEventListener('submit', async (e) => {
                    e.preventDefault();
                    const input = document.getElementById('file-input');
                    const file = input.files[0];
                    const formData = new FormData();
                    formData.append('file', file);

                    const response = await fetch('/predict/', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    document.getElementById('result').innerText = 
                        "Predicted digit: " + data.prediction;
                });
            </script>
        </body>
    </html>
    """)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("L")  # grayscale

    # Invert colors if needed
    if np.mean(np.array(image)) > 127:  # mostly white background
        image = ImageOps.invert(image)

    # Crop to bounding box of digit
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)

    # Resize to 8x8
    image = image.resize((8, 8), Image.Resampling.LANCZOS)

    # Convert to numpy array and normalize to 0-16 like load_digits
    data = np.array(image)
    data = (16 - (data / data.max()) * 16).astype(np.int32)

    # Flatten to 1D vector
    data = data.reshape(1, -1)

    prediction = model.predict(data)[0]
    return {"prediction": int(prediction)}
