from flask import Flask, request, jsonify
from waitress import serve
from asgiref.wsgi import WsgiToAsgi
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import io
import base64
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class DentalDiseaseCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(DentalDiseaseCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))  # Reduces to 1x1 feature map
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x
    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model
model = DentalDiseaseCNN(num_classes=7).to(device)
# Load the trained PyTorch model
MODEL_PATH = "dental_model.pth"
try:
    MODEL_PATH = "dental_model.pth"
    state_dict = torch.load(MODEL_PATH, map_location=device)
    
    # Print model keys
    print("Model keys:", model.state_dict().keys())

    # Print state_dict keys
    print("Checkpoint keys:", state_dict.keys())

    # Load state_dict into the model
    model.load_state_dict(state_dict,strict=False)
    model.eval()

    print("✅ PyTorch Model loaded successfully!")
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

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def preprocess_image(image: Image.Image):
    """Convert PIL image to tensor for PyTorch model."""
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def encode_image(image_array):
    """Convert OpenCV image to base64 format."""
    _, buffer = cv2.imencode(".png", image_array)
    encoded_image = base64.b64encode(buffer).decode("utf-8")
    return encoded_image

def generate_gradcam(image_tensor, predicted_class_idx):
    """Generate Grad-CAM heatmap."""
    model.eval()
    
    # Forward hook to get feature maps
    def forward_hook(module, input, output):
        global feature_maps
        feature_maps = output

    # Backward hook to get gradients
    def backward_hook(module, grad_input, grad_output):
        global gradients
        gradients = grad_output[0]

    # Register hooks on the last convolutional layer
    last_conv_layer = model.conv_layers[-3]  # Adjust based on your model architecture
    handle_f = last_conv_layer.register_forward_hook(forward_hook)
    handle_b = last_conv_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(image_tensor)
    model.zero_grad()
    
    # Backward pass to get gradients
    class_score = output[0, predicted_class_idx]
    class_score.backward()

    # Remove hooks
    handle_f.remove()
    handle_b.remove()

    # Compute Grad-CAM
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])  # Average across width & height
    for i in range(feature_maps.shape[1]):
        feature_maps[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(feature_maps, dim=1).squeeze().cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)  # ReLU to remove negatives
    heatmap /= np.max(heatmap)  # Normalize

    return heatmap

def overlay_heatmap(image, heatmap):
    """Overlay the heatmap on the original image."""
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)  # Convert to 0-255 scale
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply color map

    superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)  # Blend images
    return superimposed_img

@app.route("/api/predict/", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image_tensor = preprocess_image(image)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class_idx].item()

        predicted_disease = CLASS_NAMES[predicted_class_idx]
        print("Confidence:", confidence)

        condition = DISEASE_CONDITIONS[predicted_disease]

        
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
       
        # Generate Grad-CAM heatmap
        heatmap = generate_gradcam(image_tensor, predicted_class_idx)
        heatmap_overlay = overlay_heatmap(image_cv, heatmap)
        heatmap_encoded = encode_image(heatmap_overlay)

        return jsonify({
            "predicted_disease": predicted_disease,
            "confidence": f"{int(confidence * 100)}%",
            "condition": condition["condition"],
            "advice": condition["advice"],
            "gradcam_base64": heatmap_encoded
        })

    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8000)

