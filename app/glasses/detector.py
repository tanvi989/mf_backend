import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image

class GlassesDetector:
    def __init__(self, model_path):
        # Device FIRST to avoid errors
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model second
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Same preprocessing as training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def _load_model(self, path):
        model = resnet18(weights=None)     # Better than torch.hub
        model.fc = nn.Linear(512, 2)

        state = torch.load(path, map_location=self.device)
        model.load_state_dict(state)

        return model

    def predict(self, image: Image.Image) -> dict:
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)

            no_glasses_prob = float(probs[0, 0])
            glasses_prob = float(probs[0, 1])

        glasses_detected = glasses_prob > 0.5

        return {
            "glasses_detected": glasses_detected,
            "confidence": round(glasses_prob if glasses_detected else no_glasses_prob, 3)
        }


# Create global instance
detector = GlassesDetector("app/models/glasses_detector_resnet18.pth")
