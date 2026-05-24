import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from transformers import AutoModel
from torchvision import transforms
from PIL import Image
import cv2 as cv
import numpy as np

backbone = "facebook/dinov3-vits16-pretrain-lvd1689m"      
dino_backbone = AutoModel.from_pretrained(backbone, device_map="auto", output_hidden_states=True) #access to hiddenlayers required for FPN


#sizing down the 1280 by 720 image, whilst keeping the aspect ratio
PreProcess = transforms.Compose([
    transforms.Resize((352, 640)),      #dino eats images with the muliples of 14, 294:168 keeps the 16:9 
    transforms.ToTensor(),              #294/14 = 21 ||| 168/14 = 12        => 21 x 12 =252 + 1 patches in total
    transforms.Normalize([0.485, 0.456, 0.406], #ImageNet's mean and std...
                         [0.229, 0.224, 0.225])
])



def ImageTransformation(realsense_frames):
    #cheching whether the the input is an Image type
    if not isinstance(realsense_frames, Image.Image):   #NOTE: transformers operate only with PIL images...
        realsense_frames = Image.fromarray(realsense_frames)
    tensor = PreProcess(realsense_frames).unsqueeze(0)      #making it into a batch
    tensor = tensor.to(device)      #using cuda
    return tensor       #[Batch_size(==1), channels, height, width]


class FPN(nn.Module):
    def __init__(self, backbone):
        super(FPN, self).__init__()
        self.backbone = backbone
        self.input_h = 352
        self.input_w = 640      #NOTE: if this reso fails, try 560 × 336  (40 × 24 patches)
        self.patch_size = 16

        # freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False
        # keep backbone in eval mode (recommended for stability)
        self.backbone.eval()

        #Lateral connections: => 1 by 1 convolution, decreasing the output of FBB to specific size (e.g. 256)
        self.lat_conn1 = nn.Sequential(
            nn.Conv2d(384, 256, 1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True)   #non-linearity
        )
        self.lat_conn2 = nn.Sequential(
            nn.Conv2d(384, 256, 1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True)
        )
        self.lat_conn3 = nn.Sequential(
            nn.Conv2d(384, 256, 1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True)
        )

        #Top-down layers
        self.topdown1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True)
        )
        self.topdown2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True)
        )
        # --- smoothing after merges ---                    NOTE: NEW SCHAU MAMAL :))
        self.smooth1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True)
        )

        self.smooth2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True)
        )

        # --- fusion refinement ---                        NOTE: NEW SCHAU MAMAL :))
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True)
        )

        # Feature fusion and refinement:
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 5, 1)
        )


    def reshape(self, x):
        x = x[:, 1:, :] #we need only the spatial pathecs, so x contains only the vital datas, batch,tokens etc...
        #x = x.reshape(x.shape[0], 12, 21, x.shape[2])   #1D => 2D grid, rescalling to height(12) and width(21)
        #x = x.permute(0, 3, 1, 2) #pytorch exepcts:  [Batch, Patches,dim] -> [Batch, Height, Width, Channels]
        B, N, C = x.shape           # N == number of tokens, C == embedding dimension
        H = self.input_h // self.patch_size   # 350 / 14 = 25
        W = self.input_w // self.patch_size   # 630 / 14 = 45
        expected_tokens = H * W

        # chechking if the number of tokens 25*45 are equal with the grid positions
        if N > expected_tokens:
            x = x[:, :expected_tokens, :]
        elif N < expected_tokens:
            raise ValueError(f"Not enough tokens: {N} vs {expected_tokens}")
        x = x.reshape(B, H, W, C)
        x = x.permute(0, 3, 1, 2)    
        
        return x


    def forward(self, x):
        #the original size
        input_size = x.shape[-2:]

        #with torch.no_grad():   #we runt the image through dino without haveing to compute the gradients
        outputs = self.backbone(x, output_hidden_states=True)

        hidden_layers = outputs.hidden_states

        #outsourcing the different layer informations from dino
        c1 = self.reshape(hidden_layers[3].detach())    #low    
        c2 = self.reshape(hidden_layers[7].detach())    #mid
        c3 = self.reshape(hidden_layers[11].detach())   #high semantic informations

        del hidden_layers   #deleting to outsourced layers, to free some ram up

        #TOp-down info fusion, 
        p3 = self.lat_conn3(c3)
        #scaling up the toplayer, and adding it to the middle one, feauter blending
        p2 = self.lat_conn2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.smooth2(p2)
        #scaling up the middle one and adding it to the middle one
        p1 = self.lat_conn1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.smooth1(p1)

        # Apply topdown refinement via 3x3 convolutions
        p2 = self.topdown2(p2)
        p1 = self.topdown1(p1)
        
        #NOTE: for interpolation's mode = nearest can be also used... but bilinear should work a bit better...

        #Multiscale concatenation, upscaling every level to the highest resolution
        p2_up = F.interpolate(p2, size=p1.shape[-2:], mode="bilinear", align_corners=False)
        p3_up = F.interpolate(p3, size=p1.shape[-2:], mode="bilinear", align_corners=False)

        #placing each level next to eachother
        fusion = torch.cat([p1, p2_up, p3_up], dim=1)
        fusion = self.fusion_conv(fusion)

        #the segmentation head calculates the classes and rescales them to the orignal size
        mask_seg = self.segmentation_head(fusion)
        mask_seg = F.interpolate(mask_seg, size=input_size, mode="bilinear", align_corners=False)

        return mask_seg

device = "cuda" if torch.cuda.is_available() else "cpu"
model = FPN(dino_backbone).to(device)
model.eval()
