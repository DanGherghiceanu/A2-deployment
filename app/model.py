"""Loads the fine-tuned ResNet18 from A1 and runs single-image inference.

The preprocessing here is a deliberate copy of `eval_transform` from the A1
notebook. If the two ever drift apart, the served predictions stop matching the
evaluated model, so treat this file and the notebook's Step 3 transforms as one
unit.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models
from torchvision import transforms as T

from . import config

log = logging.getLogger("xray.model")

# Index order matches CLASS_TO_IDX in A1: NORMAL=0, PNEUMONIA=1.
CLASS_NAMES: tuple[str, str] = ("NORMAL", "PNEUMONIA")

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Identical to A1's eval_transform (resize + normalize, no augmentation).
EVAL_TRANSFORM = T.Compose(
    [
        T.Grayscale(num_output_channels=3),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested before the weights are in memory."""


class PneumoniaClassifier:
    """Wraps the fine-tuned ResNet18 checkpoint for serving."""

    def __init__(
        self,
        weights_path: Path,
        threshold: float = 0.5,
        version: str = "unknown",
    ) -> None:
        self.weights_path = Path(weights_path)
        self.threshold = float(threshold)
        self.version = version
        self.device = torch.device("cpu")
        self._model: nn.Module | None = None

    # -- lifecycle ----------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Build the architecture and load the A1 weights into it.

        `weights=None` matters: the container must not reach out to
        download.pytorch.org for ImageNet weights at startup. Every parameter
        comes from the checkpoint baked into the image.
        """
        if self._model is not None:
            return

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {self.weights_path}. "
                "Copy models/resnet18_finetuned.pt from the A1 project into this "
                "repo before building the image."
            )

        torch.set_num_threads(config.TORCH_NUM_THREADS)

        started = time.perf_counter()
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)  # single logit, as in A1

        state_dict = torch.load(self.weights_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self._model = model
        elapsed_ms = (time.perf_counter() - started) * 1000
        log.info(
            "Model loaded",
            extra={
                "json_fields": {
                    "weights_path": str(self.weights_path),
                    "size_mb": round(self.weights_path.stat().st_size / 1e6, 1),
                    "load_ms": round(elapsed_ms, 1),
                    "threshold": self.threshold,
                    "version": self.version,
                }
            },
        )

    def warmup(self) -> None:
        """Run one throwaway forward pass so the first real request isn't slow."""
        if self._model is None:
            return
        blank = Image.new("L", (IMG_SIZE, IMG_SIZE), color=0)
        self.predict(blank)
        log.info("Warmup inference complete")

    # -- inference ----------------------------------------------------------

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        """Classify one PIL image and return a JSON-serialisable result."""
        if self._model is None:
            raise ModelNotLoadedError("Model is not loaded")

        started = time.perf_counter()

        # A1 opened every image with .convert("L") before transforming.
        tensor = EVAL_TRANSFORM(image.convert("L")).unsqueeze(0).to(self.device)
        logit = self._model(tensor)
        p_pneumonia = torch.sigmoid(logit).item()

        label_index = int(p_pneumonia >= self.threshold)
        prediction = CLASS_NAMES[label_index]
        # Probability the model assigns to the class it actually chose.
        confidence = p_pneumonia if label_index == 1 else 1.0 - p_pneumonia

        return {
            "prediction": prediction,
            "label_index": label_index,
            "probability_pneumonia": round(p_pneumonia, 6),
            "confidence": round(confidence, 6),
            "class_probabilities": {
                "NORMAL": round(1.0 - p_pneumonia, 6),
                "PNEUMONIA": round(p_pneumonia, 6),
            },
            "threshold": self.threshold,
            "model_version": self.version,
            "inference_ms": round((time.perf_counter() - started) * 1000, 2),
        }


# Single instance shared across requests. Built at import time, populated with
# weights during the FastAPI lifespan startup hook.
classifier = PneumoniaClassifier(
    weights_path=config.MODEL_PATH,
    threshold=config.DECISION_THRESHOLD,
    version=config.MODEL_VERSION,
)
