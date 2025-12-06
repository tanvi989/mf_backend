import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class GlassesDetector:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load saved model weights
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Preprocessing (same as used during training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def _load_model(self, path):
        # Must match training model architecture
        model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
        model.fc = nn.Linear(512, 2)   # 2 classes → glasses / no_glasses

        # Load saved weights
        state = torch.load(path, map_location=self.device)
        model.load_state_dict(state)

        return model

    def predict(self, image: Image.Image) -> dict:
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            glasses_prob = float(probs[0][1])
            no_glasses_prob = float(probs[0][0])

        return {
            "glasses_detected": glasses_prob > 0.5,
            "confidence": round(glasses_prob if glasses_prob > 0.5 else no_glasses_prob, 3)
        }

# Create global instance
detector = GlassesDetector("app/models/glasses_detector_resnet18.pth")
