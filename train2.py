import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel
from PIL import Image
import numpy as np
import os
import random
from tqdm import tqdm

# IMPORTÁLJUK A MODELLT A MÁSIK FÁJLÉBÓL (FPN_final.py)
from FPN_final_final_05_02 import FPN 

# KONFIGURÁCIÓ 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE_ID = "facebook/dinov3-vits16-pretrain-lvd1689m" # backbone
BATCH_SIZE = 16 # elösszőr 16-al majd 32-vel érdemes megprobalni
LEARNING_RATE = 5e-5
NUM_EPOCHS = 30

TRAIN_IMG_DIR = r"C:\Users\Redux\Desktop\code\train_rgb_new"
TRAIN_MASK_DIR = r"C:\Users\Redux\Desktop\code\train_masks_new"
VAL_IMG_DIR = r"C:\Users\Redux\Desktop\code\VALIDATE_IMAGE"
VAL_MASK_DIR = r"C:\Users\Redux\Desktop\code\VALIDATE_MASK"

# ADATKEZELÉS 
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, is_train=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.is_train = is_train
        
        self.img_names = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

        self.classes = {        #detectálandó classok
            "dirt": 1,
            "grass": 2,
            "stone": 3,
            "gravel": 4
        }

        # --- MASZKOK ELŐRE INDEXELÉSE ---
        self.mask_files = os.listdir(mask_dir)
        self.image_to_masks = {}

        for img_name in self.img_names:
            base = os.path.splitext(img_name)[0]
            self.image_to_masks[base] = [m for m in self.mask_files if m.startswith(base)]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        base_name = os.path.splitext(img_name)[0]

        img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")

        # kép méret lekérdezése
        W, H = img.size
        final_mask = np.zeros((H, W), dtype=np.uint8)   #creating a mask with the same dimensions as the orig. pic.

        # végigmegyünk az összes mask file-on ami ehhez a képhez tartozik
        mask_list = self.image_to_masks[base_name]

        for mask_file in mask_list:
            for class_name, class_id in self.classes.items():
                if class_name in mask_file.lower():
                    mask_path = os.path.join(self.mask_dir, mask_file)
                    mask = np.array(Image.open(mask_path).convert("L"))
                    #NOTE: WICHTIGGG ha van 2 maszk között átfedés az egyik nem fogja felulírni a másikat!!!
                    final_mask[(mask > 128) & (final_mask == 0)] = class_id

        if self.transform:
            return self.transform(img, Image.fromarray(final_mask), train=self.is_train)

        return img, final_mask
def segmentation_transforms(image, mask, train=True):
    # MÓDOSÍTVA!!!!!!: 168x294-re cserélve, hogy passzoljon az FPN 12x21-es patch méretéhez!
    image = TF.resize(image, (352, 640))
    mask = TF.resize(mask, (352, 640), interpolation=TF.InterpolationMode.NEAREST)  #interpol. cuz the mask is a binary image...
    
    if train:       #if train is set, flip it drip it with random degree and also horizontal flips..., klein aber fein
        if random.random() > 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)   #NOTE: IMPORTANT:TO DO the same thing to the image and the corresponding mask as well, cuz theyr sorted together...
        angle = random.uniform(-15, 15)
        image, mask = TF.rotate(image, angle), TF.rotate(mask, angle)
        
    image = TF.to_tensor(image) #dino and almost everything works with tensors
    image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #dino was trained with PIL images...
    mask = torch.as_tensor(np.array(mask), dtype=torch.long)
    return image, mask


# Dice Loss implementation
def dice_loss(preds, targets, smooth=1e-6, num_classes=5):
    """
    preds: tensor of shape [B, C, H, W] (logits)
    targets: tensor of shape [B, H, W] (class indices)
    """
    preds = torch.softmax(preds, dim=1)  # convert logits to probabilities
    targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()  # [B, C, H, W]

    intersection = (preds * targets_one_hot).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)

    return 1 - dice.mean()  # return 1 - mean dice score as loss


# TANÍTÁSI ÉS VALIDÁCIÓS LÉPÉSEK
def train_epoch(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader, desc="Tanítás") #tqdm is bscly a visual ui which shows the process of the training on a colored bar...
    
    for data, targets in loop:
        data = data.to(DEVICE)
        targets = targets.to(DEVICE).long()
        
        with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        loop.set_postfix(loss=loss.item())


# This func is actually not the most important one, its only purpose is to colorize the mask, so we can merg it into the orignal image
# and have some visual results...
def colorize_mask(mask):
    colors = {
        0: (0,0,0),        # background == BLACK
        1: (139,69,19),    # dirt
        2: (0,255,0),      # grass == GREEN
        3: (128,128,128),  # stone == GREY 
        4: (255,255,0)     # gravel == WHITE
    }

    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, color in colors.items():
        color_mask[mask == cls] = color
    return color_mask

#saving only 5 images of the epoch, so it won't trash the whole memory
def save_predictions(loader, model, folder="predictions", num_images=5):
    model.eval()        #setting the mode to evaulate, so it disallows BatchNorm  and Dropout, so the predictions will be stabel
    # dropout is bscly turning a few of the neurons out, so the neural network wont just learn the pattern, sondern die echte Bilders :))
    os.makedirs(folder, exist_ok=True)  #if doesnt exists it will create a new folder...

    count = 0

    with torch.no_grad():       # it wont calculate the gradients, ==> less memory required for the operations
        for x, y in loader:     # sweeping through the batches of the dataloader
            x = x.to(DEVICE)

            preds = torch.argmax(model(x), dim=1)   # output: [batch, classes, H, W] | with argmax we calculate the best fitting class

            for i in range(preds.shape[0]):         # 
                pred = preds[i].cpu().numpy()
                color = colorize_mask(pred)         # colorizing the mask, based on the vorgegebene Farben
                Image.fromarray(color).save(f"{folder}/pred_{count}.png") #unddd zum Schluss, wir müssen es auch speichern...

                count += 1
                if count >= num_images:
                    model.train()
                    return

    model.train()


# ÚJ FÜGGVÉNY: Validáció        --> evaulating the modell with the validation datas
def check_accuracy(loader, model):
    num_correct = 0     # számoljuk, hány pixel lett helyesen előrejelzve.
    num_pixels = 0

    tp_total = torch.zeros(5).to(DEVICE)
    fp_total = torch.zeros(5).to(DEVICE)
    fn_total = torch.zeros(5).to(DEVICE)
    #átlagoláshoz, also für die Durschnittberechnung
    iou_total = 0 # IoU (Intersection over Union) -> pixel szinten mennyire fedik egymást a predikciók és a valós maszk
    batches = 0

    model.eval()        #evaulation mode

    with torch.no_grad():       #disabling the gradient calculations =>  je weniger memory verwendet,desto mehr Geschwindigkeit die CUDA hat
        for x, y in loader:     
            x = x.to(DEVICE)    
            y = y.to(DEVICE)    

            logits = model(x)   # the output of the modell...
            preds = torch.argmax(logits, dim=1)     # a modell pixelszintű predikciója, argmax-al a legvalószínűbb class-t vesszük...

            num_correct += (preds == y).sum().item()    #összesen hány pixel egyezik meg a valós predikcióval
            num_pixels += torch.numel(preds)            #össze pixel
            #kiszámoljuk osztályonként a teljesítménymutatókat, majd átlagolja őket
            tp, fp, fn = compute_confusion(preds, y, num_classes=5)      # pontossag osztályokra való lebontása
            tp_total += tp
            fp_total += fp
            fn_total += fn
        
        iou_per_class = tp_total / (tp_total + fp_total + fn_total + 1e-8)
        iou_avg = iou_per_class.mean().item()

        accuracy = num_correct / num_pixels * 100

        print(f"Accuracy: {accuracy:.2f}%")
        print(f"IoU: {iou_avg:.3f}")

    accuracy = num_correct / num_pixels * 100   

    #die Ausschrieibung den Durschnitten
    #print(f"Accuracy: {accuracy:.2f}%")
    #print(f"Precision: {precision_avg:.3f}")
    #print(f"Recall: {recall_avg:.3f}")
    #print(f"IoU: {iou_avg:.3f}")
    #train módba való visszakapcsolás
    model.train()

def compute_confusion(preds, labels, num_classes=5):
    preds = preds.view(-1)
    labels = labels.view(-1)

    device = preds.device

    tp = torch.zeros(num_classes, device=device)
    fp = torch.zeros(num_classes, device=device)
    fn = torch.zeros(num_classes, device=device)

    for cls in range(num_classes):
        tp[cls] = ((preds == cls) & (labels == cls)).sum()
        fp[cls] = ((preds == cls) & (labels != cls)).sum()
        fn[cls] = ((preds != cls) & (labels == cls)).sum()

    return tp, fp, fn


# FŐFÜGGVÉNY --> the spine of the training && validation && prediction && checkpoints :DD
def main():
    print(f"Modell betöltése: {BACKBONE_ID}...")
    
    # backbone betöltése
    dino_backbone = AutoModel.from_pretrained(BACKBONE_ID, output_hidden_states=True)
    # freeizing the backbone, => only the head is getting taught!!!
    for param in dino_backbone.parameters():
        param.requires_grad = False

    #calling FPN nn && using "cuda"
    model = FPN(dino_backbone).to(DEVICE)
    # BatchNorm stabilitás fix (Ha maradt volna a hálózatban, a GroupNorm-ot nem bántja)
    
    #class súlyok a loss-hoz 
    class_weights = torch.tensor([1.0, 2.0, 2.0, 2.0, 2.0]).to(DEVICE)      #NOTE: súlyozzuk a CROSSENTROPYLOSS-t hogy a háttér pixelek(0,0,0) ne domináljanak,
    # a kiebb osztályokra nagyobb büntetés jár!!
    
    #LOSSS FUNCTION COMBINED DICE + CROSSENTROPY LOSS

    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def combined_loss(preds, targets):
        ce_loss = ce_loss_fn(preds, targets)
        d_loss = dice_loss(preds, targets)
        return ce_loss + d_loss  # you can weight them if you want: 0.5*ce + 0.5*dice


    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)       # == tanulási algoritmus :))
    scaler = torch.amp.GradScaler(device="cuda") #GradScaler->gyorsabb + kevesebb memória
    
    # Adatok előkészítése (Tanító)      
    train_ds = SegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, segmentation_transforms, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)

    #number of workers := több szálon megy az adatok betöltése :D
        
    # ÚJ: Adatok előkészítése (Validációs)
    val_ds = SegmentationDataset(VAL_IMG_DIR, VAL_MASK_DIR, segmentation_transforms, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)
    
    print("Tanítás indul...")
    for epoch in range(NUM_EPOCHS):     # running till all of the epochs went through 
        print(f"\n=== Epoch {epoch+1}/{NUM_EPOCHS} ===")
        
        # 1. Tanít
        train_epoch(train_loader, model, optimizer, combined_loss, scaler)
        
        # 2. Ellenőrizzük, mit tanult (Validáció)
        check_accuracy(val_loader, model)
        
        save_predictions(val_loader, model, folder=f"pred_epoch_{epoch}", num_images=3)

        # 3. Mentés, mentjük az optimizer állapotát így később visszaállítható
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, f"rc_car_fpn_epoch_{epoch}.pth.tar")

if __name__ == "__main__":
    main()