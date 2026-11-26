from typing import *
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image


# Background-removal weights that are licensed for non-commercial use only
# (e.g. briaai/RMBG-2.0 is CC BY-NC 4.0). If a pipeline config asks for one of
# these, fall back to the MIT-licensed BiRefNet weights instead.
COMMERCIAL_SAFE_MODEL = "ZhengPeng7/BiRefNet"
NON_COMMERCIAL_PREFIXES = ("briaai/",)


class BiRefNet:
    def __init__(self, model_name: str = COMMERCIAL_SAFE_MODEL):
        if model_name.lower().startswith(NON_COMMERCIAL_PREFIXES):
            print(f"[BiRefNet] '{model_name}' is licensed for non-commercial use only; "
                  f"using '{COMMERCIAL_SAFE_MODEL}' (MIT) instead.")
            model_name = COMMERCIAL_SAFE_MODEL
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    def to(self, device: str):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
    