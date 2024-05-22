# TO DO:
# It would be better, to change in model defining files self.model to self.dino to emphasize that it is a DINO part.





# Imports
import torch

from typing import Literal

from Biomarker_SSL import Biomarker_Model
from End2End_SSL import End2End_Model
from utils import *



# MAIN

def main(PATIENT: str, SLICE: int, TASK: Literal["Regression", "Classification"]):
    """
    Function running XMED pipeline.
    """
    # Loading model and registering hooks:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if TASK == "Regression":
        MODEL_NR=4 # Best checkpoint will be chosen.
        biom_model = Biomarker_Model.load_from_checkpoint(
            f"./checkpoints/Biomarkers/best-checkpoint_{MODEL_NR}.ckpt").to(device).eval()
        model = biom_model.model
    else:
        MODEL_NR=4 # Best checkpoint will be chosen.
        E2E_model = End2End_Model.load_from_checkpoint(
            f"./checkpoints/End2End/best-checkpoint_{MODEL_NR}.ckpt").to(device).eval()
        model = E2E_model.model
    
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

    final_block_norm1 = model.blocks[-1].norm1
    activation_hook = final_block_norm1.register_forward_hook(
        get_activation("last_att_in"))
    grad_hook = final_block_norm1.register_full_backward_hook(
        get_gradient("last_att_in"))

    # Loading image from repository:
    ...

    # Model inference:
    model = model.to(device)
    attention_map, class_attention_map = get_maps(model, IMAGE, target_class=TARGET_CLASS, clip=True)

    



    return +1



if __name__ == "__main__":
    main()