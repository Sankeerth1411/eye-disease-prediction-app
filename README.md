# Eye Disease Prediction System

A simple Flask + TensorFlow/Keras web app that predicts eye diseases from
uploaded retinal (fundus) images.

## Project Structure

```
project/
│
├── app.py                 # Flask backend (routes, model loading, inference)
├── eye_disease_model.keras  # <-- PLACE YOUR TRAINED MODEL FILE HERE
├── requirements.txt
├── static/
│   ├── style.css
│   ├── script.js
│   └── logo.png           # replace with your own logo if you like
├── templates/
│   └── index.html
└── uploads/                # temp folder (currently unused for storage,
                             # kept for future use / logging)
```

## 1. Installation

```bash
# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Add Your Model

Copy your trained `.keras` file into the project root and name it
`eye_disease_model.keras` (or update `MODEL_PATH` at the top of `app.py`
if you'd rather keep a different filename).

**No other code changes are required** as long as:
- The model accepts a batch of RGB images and outputs a softmax vector
  (one probability per class).
- You update `IMG_SIZE` and `CLASS_NAMES` in `app.py` to match how the
  model was trained (see below).

## 3. Update Class Names

In `app.py`, edit the `CLASS_NAMES` list so it matches **the exact order**
your model was trained/exported with (e.g. the order from
`train_ds.class_names` if you used `image_dataset_from_directory`):

```python
CLASS_NAMES = [
    "Diabetic Retinopathy",
    "Glaucoma",
    "Healthy",
    "Macular Scar",
    "Myopia",
]
```

If you add or remove a class, also update the matching entry in the
`DISEASE_INFO` dictionary (description, symptoms, precautions, treatment).

## 4. Update Preprocessing (if needed)

`app.py` currently resizes images to `IMG_SIZE` (default `224x224` for
EfficientNetB0) and rescales pixel values to `[0, 1]`. If your model was
trained with a different preprocessing function (e.g.
`tf.keras.applications.efficientnet.preprocess_input`), update the
`preprocess_image()` function accordingly — mismatched preprocessing is
the most common cause of incorrect predictions after deployment.

## 5. Run the App

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

You can also check `http://localhost:5000/health` to confirm the model
loaded correctly and see the active class list / input size.

## 6. Usage

1. Click the upload area and select a retinal/fundus image.
2. Preview the image.
3. Click **Predict**.
4. View the predicted disease, confidence score, description, symptoms,
   precautions, and treatment guidance.

## Notes

- Max upload size is 8 MB (configurable via `MAX_CONTENT_LENGTH` in `app.py`).
- Allowed file types: png, jpg, jpeg, bmp, webp.
- This tool is for educational/research purposes only and is **not** a
  substitute for professional medical diagnosis.
