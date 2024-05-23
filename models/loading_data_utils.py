import torch
from torchvision import transforms
import copy
from typing import Tuple


def load_img(crop_path:str,
             MEAN:torch.Tensor,
             STD:torch.Tensor, 
             crop_view:str="axial", 
             slice_:int=16,
             device:str="cuda")->Tuple[torch.Tensor, torch.Tensor]:
    """Returns the image as resized and normalized tensor and original (only resized)"""

    channels_mean = [MEAN for i in range(3)]
    channels_std = [STD for i in range(3)]

    resizeANDnorm = transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=channels_mean, std=channels_std)
        ]
    )
    
    resize = transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True)
        ]
    )
    
    crop = torch.load(crop_path).float()
    
    if crop_view == "axial":
        img = crop[:, :, slice_]
        mask = crop[:, :, slice_]
        
    elif crop_view == "coronal":
        img = img[:, slice_, :]
        mask = mask[:, slice_, :]
        
    elif crop_view == "sagittal":
        img = img[slice_, :, :]
        mask = mask[slice_, :, :]
    
    img = torch.clamp(img, -1000, 400)
    
    if (len(img.shape) < 3):
        img = img.unsqueeze(0)
    img = img.repeat(3,1,1)
    
    original_img = copy.deepcopy(img)
    # to bring original img in [0, 1] range.
    original_img -= -1000
    original_img = original_img/1400

    img = resizeANDnorm(img).to(device)
    # Moving channels dimension.
    original_img = torch.movedim(resize(original_img).cpu(), 0, 2)

    # make image divisible by patch size
    PATCH_SIZE=8 # Dino patch size.
    w, h = (
        img.shape[1] - img.shape[1] % PATCH_SIZE,
        img.shape[2] - img.shape[2] % PATCH_SIZE,
    )
    img = img[:, :w, :h].unsqueeze(0)
    img.requires_grad = True
    return (img, original_img)