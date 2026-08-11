"""
Eye Disease Prediction System - Flask Backend
================================================

HOW TO REPLACE THE MODEL:
1. Drop your .keras file in the project root (same folder as this file).
2. Update MODEL_PATH below if the filename is different.
3. Update IMG_SIZE if your model was trained on a different input resolution
   (EfficientNetB0 typically uses 224x224 - change this if yours differs).
4. Update CLASS_NAMES to match the exact order your model outputs
   (this must match the order used during training, e.g. from
   train_ds.class_names if you used image_dataset_from_directory).
5. Update DISEASE_INFO if you add/remove/rename classes.

Nothing else in this file needs to change - the /predict route reads
these config values dynamically.
"""

import os
import io
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf

# ----------------------------------------------------------------------
# CONFIG - edit these values when you swap in your own model
# ----------------------------------------------------------------------

MODEL_PATH = "best_model.keras"
# Alternative provided file: "eye_disease_model.keras" - both share the same
# EfficientNetB0 architecture (224x224 input, 5-class output). Swap the
# string above to switch between them.

# Input size expected by the model (width, height). EfficientNetB0 default
# is 224x224 - change this if your model was trained differently.
IMG_SIZE = (224, 224)

# Class order MUST match the order the model was trained/exported with.
CLASS_NAMES = [
    "Diabetic Retinopathy",
    "Glaucoma",
    "Healthy",
    "Macular Scar",
    "Myopia",
]

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB upload limit

# ----------------------------------------------------------------------
# Disease information dictionary - one entry per class in CLASS_NAMES
# ----------------------------------------------------------------------

DISEASE_INFO = {
    "Diabetic Retinopathy": {
        "description": "Damage to retinal blood vessels caused by prolonged high blood sugar levels associated with diabetes.",
        "symptoms": ["Blurred vision", "Floaters", "Vision loss", "Difficulty seeing at night"],
        "precautions": ["Control blood sugar levels", "Regular eye examinations", "Maintain a healthy diet", "Monitor blood pressure"],
        "treatment": ["Laser therapy (photocoagulation)", "Anti-VEGF eye injections", "Vitrectomy surgery in severe cases"],
    },
    "Glaucoma": {
        "description": "A group of eye diseases that damage the optic nerve, often due to elevated intraocular pressure.",
        "symptoms": ["Tunnel vision", "Eye pain", "Blurred vision", "Halos around lights"],
        "precautions": ["Routine eye checkups", "Monitor eye pressure regularly", "Avoid activities that increase eye pressure"],
        "treatment": ["Prescription eye drops", "Laser treatment (trabeculoplasty)", "Surgical intervention"],
    },
    "Healthy": {
        "description": "No major retinal abnormalities detected in the uploaded image.",
        "symptoms": ["Normal retinal appearance", "Clear optic disc and macula"],
        "precautions": ["Regular annual eye checkups", "Maintain a healthy lifestyle", "Protect eyes from UV exposure"],
        "treatment": ["No treatment required"],
    },
    "Macular Scar": {
        "description": "Scarring in the macular region of the retina that affects sharp, central vision.",
        "symptoms": ["Blurred central vision", "Distorted vision", "Difficulty reading or recognizing faces"],
        "precautions": ["Protect eyes from further injury", "Regular monitoring by a specialist", "UV protection"],
        "treatment": ["Specialist consultation", "Vision rehabilitation", "Low-vision aids"],
    },
    "Myopia": {
        "description": "Nearsightedness - a refractive error causing distant objects to appear blurry.",
        "symptoms": ["Difficulty seeing distant objects", "Eye strain", "Headaches", "Squinting"],
        "precautions": ["Regular vision testing", "Limit prolonged near-screen exposure", "Take breaks during close-up work"],
        "treatment": ["Corrective glasses", "Contact lenses", "Refractive surgery (e.g. LASIK)"],
    },
}

# ----------------------------------------------------------------------
# App setup
# ----------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model once at startup rather than per-request.
model = None
model_load_error = None
try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[startup] Loaded model from '{MODEL_PATH}'")
    else:
        model_load_error = (
            f"Model file '{MODEL_PATH}' not found. Place your .keras file "
            f"in the project root, or update MODEL_PATH in app.py."
        )
        print(f"[startup] WARNING: {model_load_error}")
except Exception as exc:  # noqa: BLE001
    model_load_error = f"Failed to load model: {exc}"
    print(f"[startup] ERROR: {model_load_error}")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess a PIL image exactly as the model expects.

    NOTE: If your training pipeline used a specific preprocess_input
    function (e.g. tf.keras.applications.efficientnet.preprocess_input),
    swap the normalization line below to match it exactly - mismatched
    preprocessing is the #1 cause of wrong predictions after deployment.
    """
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    array = tf.keras.utils.img_to_array(image)

    # IMPORTANT: this model's EfficientNetB0 backbone has Rescaling +
    # Normalization layers baked directly into the model graph (confirmed by
    # inspecting the provided .keras files). That means the model expects
    # RAW pixel values in [0, 255] - do NOT divide by 255 or call
    # preprocess_input() here, either would double-normalize and produce
    # incorrect predictions. If you later swap in a model that does NOT
    # include internal rescaling, add the appropriate normalization here.
    array = np.expand_dims(array, axis=0)  # add batch dimension
    return array


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """Simple health check - useful to confirm the model loaded correctly."""
    return jsonify({
        "status": "ok" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "error": model_load_error,
        "classes": CLASS_NAMES,
        "img_size": IMG_SIZE,
    })

print("Prediction route reached")
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": model_load_error or "Model is not loaded on the server."
        }), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use form field name 'image'."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        processed = preprocess_image(image)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Could not process image: {exc}"}), 400

    try:
        predictions = model.predict(processed, verbose=0)[0]
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Model inference failed: {exc}"}), 500

    # Guard against a mismatch between model output size and CLASS_NAMES.
    if len(predictions) != len(CLASS_NAMES):
        return jsonify({
            "error": (
                f"Model outputs {len(predictions)} classes but CLASS_NAMES "
                f"has {len(CLASS_NAMES)} entries. Update CLASS_NAMES in app.py."
            )
        }), 500

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    info = DISEASE_INFO.get(predicted_class, {
        "description": "No additional information available for this class.",
        "symptoms": [],
        "precautions": [],
        "treatment": [],
    })

    # Full probability breakdown, useful for the frontend or debugging.
    all_probabilities = {
        CLASS_NAMES[i]: round(float(predictions[i]) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    return jsonify({
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "description": info["description"],
        "symptoms": info["symptoms"],
        "precautions": info["precautions"],
        "treatment": info["treatment"],
        "all_probabilities": all_probabilities,
    })

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
