import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.model_path = os.path.join("artifacts", "training", "model.h5")
        self.image_size = (224, 224)

    def predict(self):
        # Load model
        model = load_model(self.model_path)

        # Load and preprocess image
        test_image = image.load_img(self.filename, target_size=self.image_size)
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # ✅ Match training

        # Predict
        probs = model.predict(test_image)[0]
        pred_class = np.argmax(probs)

        # Class map (ensure it's consistent with train generator)
        class_map = {0: "Adenocarcinoma Cancer", 1: "Normal"}

        print(f"Raw Probabilities: {probs}")
        print(f"Predicted Class: {pred_class} → {class_map[pred_class]}")

        return [{
            "prediction": class_map[pred_class],
            "confidence": float(np.max(probs)),
            "class_distribution": {
                "Adenocarcinoma Cancer": float(probs[0]),
                "Normal": float(probs[1])
            }
        }]
