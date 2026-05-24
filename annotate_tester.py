from PIL import Image
import torch
import torchvision.transforms.functional as TF
import numpy as np
from FPN_final_final_05_02 import FPN
from transformers import AutoModel

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
CHECKPOINT_PATH = r"C:\Users\Redux\Desktop\code\rc_car_fpn_epoch_29.pth.tar"  # load your best epoch
IMAGE_PATH = r"C:\Users\Redux\Desktop\code\VALIDATE_IMAGE\color_000005.jpg"
ALPHA = 0.5  # transparency for overlay

# --- LOAD MODEL ---
dino_backbone = AutoModel.from_pretrained(BACKBONE_ID, output_hidden_states=True)
model = FPN(dino_backbone).to(DEVICE)

# Load weights
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# --- FUNCTIONS ---
def predict_image(img_path):
    """Preprocess image and run model prediction."""
    img = Image.open(img_path).convert("RGB")
    img_resized = TF.resize(img, (352, 640))
    img_tensor = TF.to_tensor(img_resized)
    img_tensor = TF.normalize(img_tensor, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy()
    return pred, img  # return original image too

def colorize_mask(mask):
    """Convert label mask to RGB colors for visualization."""
    colors = {
        0: (0,0,0),       # background
        1: (139,69,19),   # dirt
        2: (0,255,0),     # grass
        3: (128,128,128), # stone
        4: (255,255,0)    # gravel
    }
    h, w = mask.shape
    color_mask = np.zeros((h,w,3), dtype=np.uint8)
    for cls, color in colors.items():
        color_mask[mask==cls] = color
    return color_mask

def overlay_mask_on_image(image, mask, alpha=0.5):
    """Overlay a colored mask on top of the original image."""
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)
    mask = mask.resize(image.size)
    blended = Image.blend(image, mask, alpha)
    return blended

# --- MAIN ---
if __name__ == "__main__":
    pred_mask, original_img = predict_image(IMAGE_PATH)
    pred_color = colorize_mask(pred_mask)   

    # Show mask alone
    Image.fromarray(pred_color).show(title="Predicted Mask")

    # Overlay mask on original image
    overlayed = overlay_mask_on_image(original_img, pred_color, alpha=ALPHA)
    overlayed.show(title="Overlayed Image")

    # Optional: save results
    pred_save_path = "predicted_mask.png"
    overlay_save_path = "overlayed_result.png"
    Image.fromarray(pred_color).save(pred_save_path)
    overlayed.save(overlay_save_path)
    print(f"Saved predicted mask -> {pred_save_path}")
    print(f"Saved overlayed image -> {overlay_save_path}")