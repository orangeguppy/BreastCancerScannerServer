import torch

import helper_functions
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import mlflow
import io

from flask import Flask, request, jsonify
from PIL import Image

def process_image(image):
    # Define a transform for the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # resize all images to 224px x 224px
        transforms.ToTensor()  # convert to a tensor
    ])

    # Convert the FileStorage object to a PIL Image
    pil_image = Image.open(io.BytesIO(image.read()))
    transformed_image = transform(pil_image)

    # Add an additional dimension to represent the batch
    transformed_image = transformed_image.unsqueeze(0)
    return transformed_image

def densenet201_prediction(input):
    # Fetch and load the model
    model_uri = 'dbfs:/databricks/mlflow-tracking/596982996032551/ad4bc17d71d3430a8c88dd0441c51882/artifacts/densenet201_models'
    loaded_model = mlflow.pytorch.load_model(model_uri, map_location='cpu')

    # loaded_model.eval()  # Set the model to evaluation mode
    output = loaded_model(input)
    class_probabilities = torch.softmax(output, dim=1)
    predicted_result = (class_probabilities[0][1] >= 0.5).int()

    if (predicted_result == 0):
        return "Benign"
    else:
        return "Malignant"

app = Flask(__name__)
@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/densenet201', methods=['POST'])
def prediction():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided.'}), 400
        image = request.files['image']
        if image:
            actual_result = "Benign" if (str(image)[4] == "B") else "Malignant"
            transformed_image = process_image(image)
    except Exception as e:
        return jsonify({'error': 'An error occurred.', 'details': str(e)}), 500

    # Make predictions
    densenet201_pred = densenet201_prediction(transformed_image)

    results = {
        "Actual": actual_result,
        "DenseNet201": densenet201_pred
    }

    return results

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)