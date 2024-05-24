import torch
from torchvision import transforms
import copy
from typing import Tuple, Literal



def load_img(crop_path:str,
             MEAN:torch.Tensor=torch.Tensor([0]),
             STD:torch.Tensor=torch.Tensor([0]), 
             crop_view:Literal["axial", "coronal", "sagittal"]="axial", 
             slice_:int=16,
             return_both:bool=True,
             device:Literal["cpu", "cuda"]="cuda")->Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor]:
    """
    Function returning preprocessed nodule images (as Tensors),
    both original (resized&rescaled) and normalized, or only original.
    """
    
    crop = torch.load(crop_path).float()
    
    if crop_view == "axial":
        original_img = crop[:, :, slice_]
        mask = crop[:, :, slice_]
        
    elif crop_view == "coronal":
        original_img = crop[:, slice_, :]
        mask = mask[:, slice_, :]
        
    elif crop_view == "sagittal":
        original_img = crop[slice_, :, :]
        mask = mask[slice_, :, :]
    
    original_img = torch.clamp(original_img, -1000, 400)
    
    if (len(original_img.shape) < 3):
        original_img = original_img.unsqueeze(0)
    original_img = original_img.repeat(3,1,1)

    if return_both:
        # To return normalized image as input to model.
        norm_img = copy.deepcopy(original_img)

        channels_mean = [MEAN for _ in range(3)]
        channels_std = [STD for _ in range(3)]

        resizeANDnorm = transforms.Compose(
            [
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize(mean=channels_mean, std=channels_std)
            ]
        )
        norm_img = resizeANDnorm(norm_img).to(device)
        norm_img.requires_grad = True
        norm_img = norm_img.unsqueeze(0)

    
    # Original image is not normalized with mean and std, but is rescaled to [0,1] and resized.
    resize = transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True)
        ]
    )
    
    # to bring original img in [0, 1] range.
    original_img -= -1000
    original_img = original_img/1400

    # Moving channels dimension.
    original_img = torch.movedim(resize(original_img).cpu(), 0, 2)

    if return_both:
        return (norm_img, original_img)
    else: 
        return (original_img)