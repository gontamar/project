import torch
import numpy as np
import cv2
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification
import os
 
# Offline flags are fine
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
 
# Pick device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
 
MODEL_PATH = "models/brain_tumor"
 
class BrainEngine:
    def __init__(self):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
        self.model = SiglipForImageClassification.from_pretrained(MODEL_PATH, local_files_only=True).to(self.device)
        self.model.eval()
 
    def _to_device(self, inputs: dict):
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
 
    @torch.inference_mode()
    def classify(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = self._to_device(inputs)
        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu()
        preds = {self.model.config.id2label[i]: float(p) for i, p in enumerate(probs)}
        top_idx = int(torch.argmax(probs).item())
        return preds, image, top_idx
 
    def gradcam(self, image_pil: Image.Image, class_index: int):
        """
        NOTE: This is actually a gradient-based saliency map w.r.t. input pixels
        (post-processor), not classical Grad-CAM. Works for ViT/SigLIP and gives
        a reasonable heat overlay.
        """
        self.model.zero_grad()
 
        inputs = self.processor(images=image_pil, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
 
        
        pixel_values.requires_grad_(True)
 
        outputs = self.model(pixel_values=pixel_values)
        loss = outputs.logits[0, class_index]
        loss.backward()
 
        grads = pixel_values.grad[0]  
        grads = grads.abs().sum(dim=0)  
 
        
        grads = grads.detach().cpu().numpy()
 
 
        gmin, gmax = grads.min(), grads.max()
        if gmax - gmin < 1e-12:
            norm = np.zeros_like(grads, dtype=np.float32)
        else:
            norm = (grads - gmin) / (gmax - gmin)
 
 
        heat = cv2.resize(norm, image_pil.size, interpolation=cv2.INTER_LINEAR)
 
        # Color map
        heat_uint8 = np.uint8(255 * heat)
        heatmap = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
 
        overlay = (0.5 * heatmap + 0.5 * np.array(image_pil)).clip(0, 255).astype(np.uint8)
        return overlay
