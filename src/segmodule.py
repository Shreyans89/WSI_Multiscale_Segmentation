from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy,Precision, Recall,JaccardIndex,Dice



# Define your PyTorch Lightning module (inherits from pl.LightningModule)
class SegLightningModule(pl.LightningModule):
    def __init__(self,in_channels=12,num_classes=2, arch_name='UnetPlusPlus'
                 ,encoder_name='resnet34',crossentropy_weights=(3.0,8.0),lr=1e-3,**kwargs):
        super(SegLightningModule, self).__init__()
        # Define your model architecture here
        
        self.train_metrics = MetricCollection(prefix='Train',metrics=[
                            Accuracy(task='binary' ),
                            Precision(task='binary'),
                            Recall(task='binary'),
                           JaccardIndex(task='binary')])
        self.val_metrics = MetricCollection(prefix='Val',metrics=[
                            Accuracy(task='binary' ),
                            Precision(task='binary'),
                            Recall(task='binary'),
                           JaccardIndex(task='binary')])
        
        arch=getattr(smp, arch_name)
        self.segmentation_model=arch(encoder_name=encoder_name,in_channels=in_channels,classes=num_classes)
        self.loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(crossentropy_weights))
        self.lr=lr
        

    def forward(self, x):
        # Define the forward pass of your model
        
        return self.segmentation_model(x)
    
    def flattened_cross_entropy_loss(self,inp,tgt):
        
        tgt=tgt.flatten(start_dim=-2)
        inp=inp.flatten(start_dim=-2)
      
        return self.loss_fn(inp, tgt)
    
    
    def training_step(self, batch, batch_idx):
        # Define the training step
      
        img_b,mask_b=batch
        img_b=img_b.to(torch.float32)
        #taking the mask at the highest zoom  as target
        mask_b=mask_b[:,0,:,:].to(torch.int64) 
        
        y_pred_logits = self(img_b)
        loss = self.flattened_cross_entropy_loss(y_pred_logits, mask_b)
        y_pred=y_pred_logits.argmax(axis=1)
        
        metrics=self.train_metrics(y_pred,mask_b)
        metrics.update({'Train_loss':loss,'Train_pct_foreground':mask_b.float().mean()})
        
        
        self.log_dict(metrics,on_step=False,on_epoch=True,prog_bar=False)  # Log the training loss for TensorBoard
        return loss

    def validation_step(self, batch, batch_idx):
        # Define the training step
      
        img_b,mask_b=batch
        img_b=img_b.to(torch.float32)
        #taking the mask at the highest zoom  as target
        mask_b=mask_b[:,0,:,:].to(torch.int64) 
        
        y_pred_logits = self(img_b)
        loss = self.flattened_cross_entropy_loss(y_pred_logits, mask_b)
        y_pred=y_pred_logits.argmax(axis=1)
        
        metrics=self.val_metrics(y_pred,mask_b)
        metrics.update({'Val_loss':loss,'Val_pct_foreground':mask_b.float().mean()})
        self.log_dict(metrics,on_step=False,on_epoch=True,prog_bar=False)  # Log the training loss for TensorBoard
        return loss
        


    def configure_optimizers(self):
        # Define your optimizer
        optimizer = optim.Adam(self.parameters(),lr=self.lr)
        return optimizer


class InferenceLightningModule(pl.LightningModule):
    def __init__(self,seg_module:SegLightningModule,inference_fpath:pathlib.Path,**kwargs):
       super(SegLightningModule, self).__init__()
        # Define your model architecture here
        
       self.seg_module=seg_module
       self.inference_fpath=inference_fpath

    def forward(self, x):
        # Define the forward pass of your model
        
        return  self.seg_module(x)
    
    

    def get_inference_schema(self,sample_json='inference_schema.geojson'):
        json_path=self.inference_path/sample_json
        with open(json_path) as json_file:
             schema = json.load(json_file)

        return schema
        
    


    def validation_step(self, batch, batch_idx):
        # Define the training step
      
        img_b,top_left_b=batch
        img_b=img_b.to(torch.float32)
       
        y_pred_logits = self(img_b)
        
        y_pred=y_pred_logits.argmax(axis=1)
        pred_coords=torch.cat([torch.argwhere(pred)+top_left for pred,top_left in zip(y_pred,top_left_b) if pred.sum()>0])
     
        
       
      
        


    
