# TO DO:
# It would be better, to change in model defining files self.model to self.dino to emphasize that it is a DINO part.





# Imports
import torch
import pickle
from typing import Literal, Tuple, Dict

from models.Biomarker_SSL import Biomarker_Model
from models.End2End_SSL import End2End_Model
from models.att_cdam_utils import get_maps
from loading_data_utils import load_img

# Globals
### Normalizing MEAN:
with open("data/fitted_mean_std.pkl", 'rb') as f:
    dict_ = pickle.load(f)

MEAN = dict_["mean"]
STD = dict_["std"]


# MAIN

def model_pipeline(NODULE: str, SLICE: int, TASK: Literal["Regression", "Classification"])->Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Function running XMED pipeline.
    """
    # Loading model and registering hooks:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if TASK == "Regression":
        MODEL_NR=4 # Best checkpoint will be chosen.
        biom_model = Biomarker_Model.load_from_checkpoint(
            f"models/checkpoints/Biomarkers/best-checkpoint_{MODEL_NR}.ckpt").to(device).eval()
        model = biom_model
    else:
        MODEL_NR=4 # Best checkpoint will be chosen.
        E2E_model = End2End_Model.load_from_checkpoint(
            f"models/checkpoints/End2End/best-checkpoint_{MODEL_NR}.ckpt").to(device).eval()
        model = E2E_model
    
    ## Creating hooks:
    activation = {}
    def get_activation(name):
        """
        Function to extract activations before the last MHSA layer.
        """
        def hook(model, input, output):
            activation[name] = output[0].detach()
        return hook
    
    grad = {}
    def get_gradient(name):
        """
        Function to extract gradients.
        """
        def hook(model, input, output):
            grad[name] = output
        return hook
    
    ## Registering hooks:
    ### We store the: 
    #### i) normalized activations entering the last MHSA layer.
    #### ii) gradients wrt the normalized inputs to the final attention layer.
    ### Both are required to compute CDAM score.
    ### We don't need to register hook on MHSA to extract attention weights, because DINO backbone has it already implemented.

    final_block_norm1 = model.model.blocks[-1].norm1
    activation_hook = final_block_norm1.register_forward_hook(
        get_activation("last_att_in"))
    grad_hook = final_block_norm1.register_full_backward_hook(
        get_gradient("last_att_in"))

    # Loading image from repository:
    img, original_img = load_img(crop_path=f"cache/crops/{NODULE}.pt", 
                                 crop_view="axial", 
                                 slice_=SLICE,
                                 MEAN=MEAN,
                                 STD=STD, 
                                 device=device)

    # Model inference:
    model = model.to(device)
    attention_map, CDAM_maps, model_output = get_maps(model, img, grad, activation, TASK, clip=True) 

    return (original_img, attention_map, CDAM_maps, model_output)



