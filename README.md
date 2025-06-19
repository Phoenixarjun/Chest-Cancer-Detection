# 🩺 ChestScan AI - Chest Cancer Prediction

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey?logo=flask)](https://flask.palletsprojects.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.2.2-blue?logo=mlflow)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-enabled-purple?logo=dvc)](https://dvc.org/)
[![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/Guide/HTML/HTML5)
[![CSS3](https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/CSS)
[![jQuery](https://img.shields.io/badge/jQuery-0769AD?logo=jquery&logoColor=white)](https://jquery.com/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Project Overview

**ChestScan AI** is a complete end-to-end MLOps project that automates the detection of **Adenocarcinoma Cancer** from chest CT scan images using a deep learning model based on VGG16.

This project integrates:
- ✅ **TensorFlow** for model development  
- ✅ **MLflow** for experiment tracking and model management  
- ✅ **DVC** for dataset versioning and pipeline reproducibility  
- ✅ **Flask + HTML/CSS/jQuery** for a simple prediction UI

It’s more than a model — it’s a reproducible, scalable machine learning system built with real-world MLOps tools.

---

## ⚙️ Installation Instructions

Clone the repository:

```bash
git clone https://github.com/yourusername/Chest-Cancer-Detection.git
cd Chest-Cancer-Detection
````

Install the dependencies:

```bash
pip install tensorflow==2.12.0 pandas gdown dvc mlflow==2.2.2 notebook flask scikit-learn matplotlib opencv-python
```

---

## 🔁 Workflow Details

This project uses a YAML-config-driven modular pipeline design.

### 📂 Key Configs

* `config.yaml`: paths and artifacts
* `params.yaml`: hyperparameters (image size, batch size, etc.)
* `secrets.yaml`: (optional) for credentials
* `dvc.yaml`: pipeline orchestration

### 🧱 Steps to Modify

1. Update `config.yaml`
2. Update `params.yaml` if needed
3. Modify entities and configuration manager
4. Implement pipeline components
5. Link everything in `main.py`
6. Use `dvc repro` to run the full pipeline

---

## 🔬 Process Steps

```mermaid
graph TD
    A[01_data_ingestion] --> B[02_prepare_basemodel]
    B --> C[03_model_trainer]
    C --> D[04_model_evaluation_with_mlflow]
```

```
📁 01_data_ingestion
    ⤷ Downloads and extracts CT scan dataset

📁 02_prepare_basemodel
    ⤷ Loads pretrained VGG16, modifies top layer

📁 03_model_trainer
    ⤷ Trains model with data generators and augmentation

📁 04_model_evaluation_with_mlflow
    ⤷ Logs metrics, artifacts, and model to MLflow
```

---

## 🧠 Model Specification

| Parameter    | Value           |
| ------------ | --------------- |
| AUGMENTATION | ✅ True          |
| IMAGE\_SIZE  | `[224, 224, 3]` |
| BATCH\_SIZE  | `16`            |
| INCLUDE\_TOP | ❌ False         |
| EPOCHS       | `15`            |
| CLASSES      | `2`             |
| WEIGHTS      | `imagenet`      |

### 📊 Final Performance Metrics

```json
{
  "loss": 0.22484230995178223,
  "accuracy": 0.9791666865348816
}
```

---

## 🧰 Technology Stack

| Layer               | Tool/Tech          | Reason for Use                              |
| ------------------- | ------------------ | ------------------------------------------- |
| Frontend            | HTML, CSS, jQuery  | Lightweight, fast UI for testing            |
| Backend             | Flask              | Python-native, easy API layer               |
| Model               | TensorFlow + VGG16 | Fast convergence + transfer learning        |
| Orchestration       | DVC                | Pipeline, artifact, and dataset tracking    |
| Experiment Tracking | MLflow             | Logs params, metrics, artifacts, and models |

---

## 📈 MLflow Experiment Tracking

* Every training run is logged using **MLflow**
* Tracks:

  * Parameters
  * Metrics (loss, accuracy)
  * Models and artifacts
* UI Access:

```bash
mlflow ui
```

Visit [http://localhost:8080](http://localhost:8080) in your browser.

---

## 📦 DVC Integration

* Tracks datasets and models across versions
* Reproducible pipelines using `dvc.yaml`
* Run the full pipeline with:

```bash
dvc repro
```

* Collaborators can sync the data with:

```bash
dvc pull
```

---

## 🖼️ Screenshots

### 🔍 Model Prediction UI

![Home Page](https://github.com/user-attachments/assets/3145a457-9bdf-4caf-a5e3-d4ddbacea26f)


![Upload Section](https://github.com/user-attachments/assets/96584733-e084-496e-a44a-75d2fbac3665)


![Prediction Section - Normal](https://github.com/user-attachments/assets/a7135c18-78c9-4c7f-bd12-c246a3cfdd7a)


![Prediction Section - Adenocarcinoma Cancer](https://github.com/user-attachments/assets/7d61df54-0d79-4ffe-a01c-be748a57291b)


![Footer Section](https://github.com/user-attachments/assets/70934049-85d1-45e5-aeb3-b704321d97e6)

### 📊 MLflow Dashboard

![MLflow UI](https://github.com/user-attachments/assets/25593be4-3d5c-430b-ac82-ec24545591b2)

---

## 🔮 Future Improvements

* ✅ Add Docker support for deployment
* ✅ Enable GitHub Actions for CI/CD
* 🔍 Add Grad-CAM visualizations for explainability
* 📦 Publish to Hugging Face or TensorFlow Hub

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🙌 Credits

Created by **Naresh B A**

* 🎓 B.Tech IT, Sri Sairam Institute of Technology
* 💡 Full Stack Developer | AI/ML Enthusiast
* 🔗 [LinkedIn](www.linkedin.com/in/naresh-b-a-1b5331243)

Drop a ⭐ if you found this project helpful!

```

