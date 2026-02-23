import torch
import torchxrayvision as xrv
import skimage.io
import numpy as np
import cv2
 
 
class VisionEngine:
    def __init__(self, device=None):
 
        self.device = device or (
            torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
 
        self.model = xrv.models.DenseNet(weights="densenet121-res224-chex")
        self.model = self.model.to(self.device)
        self.model.eval()
 
        self.pathologies = list(self.model.pathologies)
 
        self.gradients = None
        self.activations = None
        self.target_layer = self.model.features.denseblock4.denselayer16.conv2
 
    def _save_gradients(self, grad):
        self.gradients = grad
 
    def _save_activations(self, module, input, output):
        self.activations = output
 
    def process_image(self, img_path):
 
        img = skimage.io.imread(img_path)
        img = xrv.datasets.normalize(img, 255)
 
        if len(img.shape) > 2:
            img = img[:, :, 0]
 
        h, w = img.shape
        size = min(h, w)
        sh, sw = (h - size) // 2, (w - size) // 2
        img = img[sh:sh + size, sw:sw + size]
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
 
        img_tensor = torch.from_numpy(img).float()
        img_tensor = img_tensor[None, None, :, :].to(self.device)
        img_tensor.requires_grad_(True)
 
        handle_act = self.target_layer.register_forward_hook(self._save_activations)
 
        outputs = self.model(img_tensor)[0]
        scores = outputs.detach().cpu().numpy().tolist()
        pathology_results = dict(zip(self.pathologies, scores))
 
        # ---- NO FINDING (explicit) ----
        max_prob = max(pathology_results.values())
        pathology_results["No Finding"] = float(1.0 - max_prob)
 
        # ---- PRIMARY RESULT (NEW) ----
        primary_label, primary_score = max(
            pathology_results.items(), key=lambda x: x[1]
        )
 
        heatmap = None
        if primary_label in self.pathologies:
            target_idx = self.pathologies.index(primary_label)
 
            self.model.zero_grad()
            handle_grad = self.activations.register_hook(self._save_gradients)
            outputs[target_idx].backward()
 
            weights = torch.mean(self.gradients, dim=(0, 2, 3))
            cam = torch.zeros(self.activations.shape[2:], device=self.device)
 
            for i, w in enumerate(weights):
                cam += w * self.activations[0, i]
 
            cam = cam.detach().cpu().numpy()
            cam = np.maximum(cam, 0)
 
            if cam.max() > 0:
                cam = cam / cam.max()
 
            heatmap = np.uint8(cam * 255)
            handle_grad.remove()
 
        handle_act.remove()
 
        return pathology_results, heatmap, primary_label, primary_score
 
    def overlay_heatmap(self, img_path, heatmap):
 
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
        h, w = img.shape[:2]
        size = min(h, w)
        sh, sw = (h - size) // 2, (w - size) // 2
        img = img[sh:sh + size, sw:sw + size]
        img = cv2.resize(img, (512, 512))
 
        if heatmap is None:
            return img
 
        heatmap = cv2.resize(heatmap, (512, 512))
        heatmap_col = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_col = cv2.cvtColor(heatmap_col, cv2.COLOR_BGR2RGB)
 
        return cv2.addWeighted(img, 0.6, heatmap_col, 0.4, 0)
 
