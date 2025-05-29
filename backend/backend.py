import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch
from PIL import Image
import time
import torchvision.models as models
import os
from torchvision import transforms, datasets


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app, support_credentials=True)



@app.route('/identify', methods=['POST'])
def predict_species():
    if 'filename' not in request.files:
        return jsonify({'error': 'No file found'}), 400
    file = request.files['filename']
    print(file)
    if file.filename == '':
        return jsonify({'error': 'No file found'}), 400

    print("Current working directory:", os.getcwd())

    filename = secure_filename(file.filename)
    file_path = os.path.join("..", "uploads", filename)
    file.save(file_path)

    try:

        print(f"Processing file: {file_path}")

        result = get_species(file_path)
        # Return result
        return jsonify({'result': result})

    finally:
        print("done")
        if os.path.exists(file_path):
            os.remove(file_path)

def get_species(filepath):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(filepath).convert('RGB')
    image = transform(image).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50()

    num_classes = 404
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(torch.load('../model/phase2.pt', map_location=device))
    model.to(device)
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        _, predicted = torch.max(output, 1)

    class_names = sorted(os.listdir("../Test"))
    print("class names", class_names)
    predicted_class = class_names[predicted.item()]

    return predicted_class


app.run(port=5001, debug=True)