from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io
import uvicorn
import base64
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
MODEL_PATH = "saved_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    print("✅ Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Define class names and disease conditions
CLASS_NAMES = ["healthy", "calculus", "caries", "gingivitis", "hypodontia", "toothDiscoloration", "ulcers"]
DISEASE_CONDITIONS = {
    "healthy": {"condition": "Good", "advice": "Your teeth look healthy! Maintain oral hygiene."},
    "calculus": {"condition": "Bad", "advice": "Visit a dentist for cleaning to prevent gum disease."},
    "caries": {"condition": "Bad", "advice": "Cavities should be treated early to avoid serious issues."},
    "gingivitis": {"condition": "Bad", "advice": "Maintain oral hygiene and visit a dentist if inflammation persists."},
    "hypodontia": {"condition": "Good", "advice": "If causing issues, consult a dentist for treatment options."},
    "toothDiscoloration": {"condition": "Good", "advice": "If cosmetic concern, whitening treatments are available."},
    "ulcers": {"condition": "Critical", "advice": "If ulcers persist for more than two weeks, consult a dentist immediately."}
}

def find_best_layer(model):
    for layer in reversed(model.layers):  # Start from the last layers
        if hasattr(layer, 'output_shape') and isinstance(layer.output_shape, tuple):
            if len(layer.output_shape) == 4:  # Ensure it's a convolutional layer
                return layer.name  # Return the name of the best layer
    return "Conv_1"  # Return None if no valid convolutional layer found


def get_gradcam(image_array, model, predicted_class):
    layer_name = find_best_layer(model)  # Dynamically find the best convolutional layer
    if not layer_name:
        raise ValueError("No valid convolutional layer found for Grad-CAM.")

    grad_model = tf.keras.models.Model(
        inputs=model.input, 
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    # Apply weighting
    for i in range(conv_outputs.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # ReLU activation

    # Normalize heatmap
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-7)

    return heatmap


def overlay_gradcam(image, heatmap, alpha=0.5):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 🔥 Keep top 50% activations instead of removing too much
    threshold = np.percentile(heatmap, 50)
    heatmap = np.where(heatmap >= threshold, heatmap, 0)

    activation_strength = compute_activation_strength(heatmap)  
    adaptive_alpha = min(0.7, max(0.3, activation_strength))  # Use activation strength for better visualization


    superimposed_img = cv2.addWeighted(image, 1 - adaptive_alpha, heatmap, adaptive_alpha, 0)   
    return superimposed_img

def compute_activation_strength(heatmap):
    return np.sum(heatmap) / np.prod(heatmap.shape)  # Measure focus strength

def preprocess_image(image: Image.Image):
    """Preprocess image for model prediction."""
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0  # Normalize
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

def encode_image(image_array):
    """Convert NumPy image array to Base64 string."""
    _, buffer = cv2.imencode(".png", image_array)
    encoded_image = base64.b64encode(buffer).decode("utf-8")
    return encoded_image

@app.post("/prediction/")
async def predict(file: UploadFile = File(...)):
    """Predict the dental disease from an uploaded image."""
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_array = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(image_array)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        predicted_disease = CLASS_NAMES[predicted_class_idx]
        print("Confidence : ",confidence)

        # ✅ Override conditions:
        # If confidence < 90% OR predicted class is not "Caries" or "Gingivitis" → Set to "Healthy"
        if confidence < 0.70 or predicted_disease not in ["caries", "gingivitis"]:
            predicted_disease = "healthy"


        # Get condition details
        condition = DISEASE_CONDITIONS[predicted_disease]
        
        heatmap = get_gradcam(image_array, model, predicted_class_idx)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gradcam_image = overlay_gradcam(image_cv, heatmap)
        gradcam_image = encode_image(gradcam_image)

        return {
            "predicted_disease": predicted_disease,
            "confidence": str(int(confidence * 100)) + "%",
            "condition": condition["condition"],
            "advice": condition["advice"],
            "gradcam_base64": gradcam_image
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)