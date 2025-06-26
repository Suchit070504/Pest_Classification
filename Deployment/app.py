import io
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import torchvision.models as models
import torch.nn as nn

app = Flask(__name__)

# Load model
class ModifiedEfficientNet(nn.Module):
    def __init__(self, base_model, num_classes=102):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(base_model.classifier[1].in_features, num_classes)
    
    def forward(self, x):
        x = self.base_model.features(x)
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Initialize model
def load_model():
    base_model = models.efficientnet_b0(pretrained=False)
    model = ModifiedEfficientNet(base_model)
    model.load_state_dict(torch.load('model/pest_classifier.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Load pesticide data
df = pd.read_csv('model/full_pesticide_recommendation.csv')

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_class = probabilities.max(1)
            predicted_class = predicted_class.item()
            confidence = confidence.item() * 100
        
        pest_info = df[df["Pest Class ID"] == predicted_class]
        result = {
            'class_id': predicted_class,
            'confidence': f"{confidence:.2f}%",
            'pest_name': pest_info['Pest Name'].values[0] if not pest_info.empty else "Unknown",
            'pesticide': pest_info['Recommended Pesticide'].values[0] if not pest_info.empty else "N/A",
            'amount': pest_info['Amount'].values[0] if not pest_info.empty else "N/A",
            'image_data': image_bytes
        }
        
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
