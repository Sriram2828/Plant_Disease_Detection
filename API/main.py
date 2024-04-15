# Importing necessary packages
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image  # used to read images
import tensorflow as tf

app = FastAPI()


Model = tf.keras.models.load_model("../Trained_models/1")
Class_Names = ["Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
               "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite",
               "Tomato__Target_Spot", "Tomato__Tomato_YellowLeaf__Curl_Virus",
               "Tomato__Tomato_mosaic_virus", "Tomato_healthy"]


# Creating an endpoint
@app.get("/ping")
async def ping():
    return "Hello CodeCrafter001"


# Function to convert the requested data and converts to a numpy array
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


# Create a entry point for model prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):  # this function name can be different
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)  # to expand the shape (by adding a dimension)
    predictions = Model.predict(img_batch)

    predicted_class = Class_Names[np.argmax(predictions[0])]

    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
