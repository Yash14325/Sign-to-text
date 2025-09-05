from flask import Flask, render_template, request, jsonify
import numpy as np
import base64
import cv2
import json
from tensorflow.keras.models import load_model
import mediapipe as mp

app = Flask(__name__)

# Load model
model = load_model("model/asl_model.h5")

# Load class indices (your JSON mapping)
with open("model/class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse mapping (index → label)
index_to_class = {v: k for k, v in class_indices.items()}

# Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_data = data["image"].split(",")[1]
        decoded = base64.b64decode(image_data)
        np_arr = np.frombuffer(decoded, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            h, w, _ = frame.shape

            # Get bounding box
            x_min = min([lm.x for lm in hand_landmarks.landmark])
            y_min = min([lm.y for lm in hand_landmarks.landmark])
            x_max = max([lm.x for lm in hand_landmarks.landmark])
            y_max = max([lm.y for lm in hand_landmarks.landmark])

            # Convert to pixel values
            x1, y1 = int(x_min * w), int(y_min * h)
            x2, y2 = int(x_max * w), int(y_max * h)

            # Add padding
            pad = 20
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

            hand_crop = frame[y1:y2, x1:x2]

            # Resize for model
            resized = cv2.resize(hand_crop, (64, 64)) / 255.0
            image_array = resized.reshape(1, 64, 64, 3)

            # Prediction
            prediction = model.predict(image_array)[0]

            # Top-3 classes
            top3_idx = prediction.argsort()[-3:][::-1]
            top3 = [
                {"label": index_to_class[i], "confidence": float(prediction[i])}
                for i in top3_idx
            ]

            # Main prediction
            predicted_class = int(np.argmax(prediction))
            predicted_label = index_to_class[predicted_class]

            response = {"prediction": predicted_label, "top3": top3}

            # Print to console for debugging
            print("Prediction result:", json.dumps(response, indent=2))

            return jsonify(response)
        else:
            return jsonify({"prediction": "🖐 Show your hand"})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
