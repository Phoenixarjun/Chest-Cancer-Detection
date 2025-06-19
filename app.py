from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from chest_cancer_detection.utils.common import decodeImage
from chest_cancer_detection.pipeline.prediction import PredictionPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        if not request.is_json:
            return jsonify({"error": "Request content-type must be application/json"}), 400

        data = request.get_json(force=True)

        if 'image' not in data:
            return jsonify({"error": "Missing 'image' field in request body"}), 400

        decodeImage(data['image'], clApp.filename)
        result = clApp.classifier.predict()

        return jsonify({
            "prediction": result[0]["prediction"],
            "confidence": result[0]["confidence"],
            "class_distribution": result[0]["class_distribution"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)