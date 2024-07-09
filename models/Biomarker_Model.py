import torch
import torchvision
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.regression import MeanSquaredError
from typing import Literal, Union, Optional

def set_encoder_dropout_p(module, dropout_p):
    if isinstance(module, nn.Dropout):
        # Sets dropout probability for dropout layers
        module.p = dropout_p


class Biomarker_Model(pl.LightningModule):
    def __init__(self,
                 model_type:Literal["dino_vits8", "dino_vitb8", "dino_vits16", "dino_vitb16", "vit_b_16", "vit_l_16"]="dino_vits8",
                 trainable_layers:Union[int, Literal["all"]]="all", 
                 backbone_dropout:float=0.0,
                 max_lr:float=5e-6,
                 steps_per_epoch:int=30,
                 epochs:int=45,
                 div_factor:int=100,
                 n_cycles:int=6
                ):
        
        super().__init__()
        self.epochs=epochs
        self.backbone_dropout=backbone_dropout
        self.max_lr=max_lr
        self.div_factor=div_factor
        self.steps_per_epoch=steps_per_epoch
        self.n_cycles=n_cycles
        self.model_type=model_type
        self.save_hyperparameters()

        if model_type in ["dino_vits8", "dino_vitb8", "dino_vits16", "dino_vitb16"]:
            self.backbone = torch.hub.load("facebookresearch/dino:main", model_type)
            hidden_size=self.backbone.embed_dim
        elif model_type in ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14']:
            self.backbone=torch.hub.load('facebookresearch/dinov2', model_type)
            hidden_size=self.backbone.embed_dim
        elif model_type=="vit_b_16":
            self.backbone=torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
            self.backbone.heads=nn.Identity(self.backbone.hidden_dim)
            hidden_size=self.backbone.hidden_dim
        elif model_type=="vit_l_16":
            self.backbone=torchvision.models.vit_l_16(weights='IMAGENET1K_V1')
            self.backbone.heads=nn.Identity(self.backbone.hidden_dim)
            hidden_size=self.backbone.hidden_dim 
        else:
            raise Exception("Provided model is not handled.")

        self.mlp_head=nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 8)
                                   )
        
        # changing backbone dropout values:
        if backbone_dropout > 0.0:       
            self.backbone.apply(lambda module: set_encoder_dropout_p(module, dropout_p=self.backbone_dropout))

        if trainable_layers != "all":
            all_layers = len(list(self.backbone.parameters()))
            for i, p in enumerate(self.backbone.parameters()):
                if i < (all_layers - trainable_layers):
                    p.requires_grad = False
        
        self.MSE = MeanSquaredError(squared=True)
        self.RMSE = MeanSquaredError(squared=False)

    def forward(self, x):
        x = self.backbone(x)
        x = self.mlp_head(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        y = y.float()
        loss = F.huber_loss(y_hat, y)
        self.log("train_mse", self.MSE(y_hat, y), prog_bar=False, on_epoch=True, on_step=False)
        self.log("train_loss", loss, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits_views = torch.zeros((3, y.shape[0], 8), device=self.device)
        for i in range(3):
            logits_views[i] = self(x[i])
        y_hat = torch.mean(logits_views, axis=0).squeeze()

        if len(y_hat.size()) == 0:
            y_hat = y_hat.unsqueeze(dim=0)
        y = y.float()
        loss = F.huber_loss(y_hat, y)
        self.log("val_mse", self.MSE(y_hat, y), prog_bar=False, on_epoch=True, on_step=False)
        self.log("val_rmse", self.RMSE(y_hat, y), prog_bar=False, on_epoch=True, on_step=False)
        self.log("val_loss", loss, prog_bar=False, on_epoch=True, on_step=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-5) # It basic, default value of LR. initial learning rate is adjusted by a scheduler.
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                         base_lr=self.max_lr/self.div_factor, 
                                                         max_lr=self.max_lr,
                                                         cycle_momentum=False,
                                                         step_size_up=0.5*(self.epochs*self.steps_per_epoch)//self.n_cycles,
                                                         step_size_down=1.5*(self.epochs*self.steps_per_epoch)//self.n_cycles,
                                                         mode="triangular2"
                                                        )
        
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step'
        }
        
        return [optimizer], [scheduler]
