import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import os

from chest_cancer_detection.utils.common import save_json
from chest_cancer_detection.entity.config_entity import EvaluationConfig

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1. / 255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode="categorical",
            shuffle=False
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()

        # Evaluate model
        self.score = self.model.evaluate(self.valid_generator)

        # Predict
        steps = self.valid_generator.samples // self.valid_generator.batch_size + 1
        y_pred_probs = self.model.predict(self.valid_generator, steps=steps)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = self.valid_generator.classes[:len(y_pred)]
        class_names = list(self.valid_generator.class_indices.keys())

        # Classification report
        self.report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        print("📊 Classification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

        # Save score + report
        self.save_score()
        self.save_classification_report()

        # Save confusion matrix
        self.save_confusion_matrix(y_true, y_pred, class_names)

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def save_classification_report(self):
        with open("classification_report.json", "w") as f:
            json.dump(self.report, f, indent=4)

    def save_confusion_matrix(self, y_true, y_pred, class_names):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})
            mlflow.log_artifact("scores.json")
            mlflow.log_artifact("classification_report.json")
            mlflow.log_artifact("confusion_matrix.png")

            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")
