"""
Chart Classifier
"""
import gc
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)

CLASS_MAPPING = {
    0: "area", 1: "bar", 2: "box", 3: "heatmap",
    4: "histogram", 5: "line", 6: "pie", 7: "scatter"
}
KEEP_CLASSES = {"bar", "pie", "line"}
NUM_CLASSES = len(CLASS_MAPPING)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Sequential(
            nn.Identity(),
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


class ChartClassifier:
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading ResNet18 from {model_path} ...")

        self.model = ResNetClassifier(NUM_CLASSES)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device).eval()

        logger.info(f"ResNet18 ready on {self.device}")

    def classify(self, image: Image.Image) -> str:
        """
        Trả về chart type: bar / pie / line / other
        """
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            top_idx = torch.argmax(probs).item()

        raw_class = CLASS_MAPPING[top_idx]
        chart_type = raw_class if raw_class in KEEP_CLASSES else "other"

        # FIX: lưu conf trước khi del probs
        conf = probs[top_idx].item()

        # Cleanup SAU KHI đã lấy hết giá trị cần dùng
        del tensor, outputs, probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        logger.info(f"Chart classified as: {chart_type} (raw: {raw_class}, conf: {conf:.3f})")
        return chart_type
