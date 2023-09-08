
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy,Precision, Recall,JaccardIndex,Dice
import torchvision.transforms as transforms


# Define your PyTorch Lightning module (inherits from pl.LightningModule)
class SegLightningModule(pl.LightningModule):
    def __init__(self,in_channels=3,num_classes=2, arch_name='UnetPlusPlus'
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
        #self.register_buffer('mask_color',torch.tensor([255,255,255]).to(torch.uint8).unsqueeze(0).unsqueeze(2).unsqueeze(3))
        

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
        metrics.update({'Train_loss':loss,'Train_pct_foreground':mask_b.float().mean(),'Train_IOU':(y_pred*mask_b).sum()/(y_pred.sum()+mask_b.sum())})

        ## save pred and ground truth masks every epoch
        ## only log 8 masks from batch
        
        self.log_dict(metrics,on_step=False,on_epoch=True,prog_bar=False)  # Log the training loss for TensorBoard

        # if batch_idx==0:
        #     grid_pred=make_grid(y_pred[:8].to(torch.int8).unsqueeze(1)*self.mask_color,nrow=1).permute(1,2,0)
        #     grid_targ=make_grid(mask_b[:8].to(torch.int8).unsqueeze(1)*self.mask_color,nrow=1).permute(1,2,0)
        #     grid=torch.cat([grid_pred,grid_targ],dim=1)
        #     plt.imsave(Path(self.logger.log_dir)/'.'.join(['Train_epoch'+str(self.current_epoch),'jpg']),grid)
            
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
        metrics.update({'Val_loss':loss,'Val_pct_foreground':mask_b.float().mean() ,'Val_IOU':(y_pred*mask_b).sum()/(y_pred.sum()+mask_b.sum())})
        self.log_dict(metrics,on_step=False,on_epoch=True,prog_bar=False)  # Log the training loss for TensorBoard
        
        # if batch_idx==0:
        #     grid_pred=make_grid(y_pred[:8].to('torch.uint8')unsqueeze(1)*self.mask_color,nrow=1).permute(1,2,0)
        #     grid_targ=make_grid(mask_b[:8].to('torch.uint8').unsqueeze(1)*self.mask_color,nrow=1).permute(1,2,0)
        #     grid=torch.cat([grid_pred,grid_targ],dim=1)
        #     plt.imsave(Path(self.logger.log_dir)/'.'.join(['Train_epoch'+str(self.current_epoch),'jpg']),grid)
            
        
        return loss
        


    def configure_optimizers(self):
        # Define your optimizer
        optimizer = optim.Adam(self.parameters(),lr=self.lr)
        return optimizer


class InferenceLightningModule(pl.LightningModule):
    def __init__(self,seg_module:SegLightningModule,
                 WSI_shape,downsample_factor=16,**kwargs):
       super(InferenceLightningModule, self).__init__()
        # Define your model architecture here
        
       self.seg_module=seg_module
     
       W,H=WSI_shape
       self.downsample_factor=downsample_factor
       ## np array like indexing is reversed of slide dims
       self.WSI_mask=np.zeros((H//downsample_factor,W//downsample_factor))
       
        

    def forward(self, x):
        # Define the forward pass of your model
        
        return  self.seg_module(x)
    
    

    def get_inference_schema(self,sample_json='inference_schema.geojson'):
        json_path=self.inference_path/sample_json
        with open(json_path) as json_file:
             schema = json.load(json_file)

        return schema
        
    


    def test_step(self, batch, batch_idx):
        # Define the training step
      
        top_left_b, img_b=batch
         
        _,_,h,w=img_b.shape
        h_ds,w_ds=tuple(np.array((h,w))//self.downsample_factor)
        
        img_b=img_b.to(torch.float32)
        y_pred_logits = self(img_b)
        y_pred=y_pred_logits.argmax(axis=1)
        y_pred_ds=transforms.Resize((h//self.downsample_factor,w//self.downsample_factor))(y_pred)
        ##  append masks for each batch
        for top_left,pred in zip(top_left_b.cpu(),y_pred_ds.cpu()):
            Y_start,X_start=tuple(np.array(top_left)//self.downsample_factor)
            try:
                self.WSI_mask[Y_start:Y_start+h_ds,X_start:X_start+w_ds]=pred
            except:
                pdb.set_trace()
            
            
            
        
        
        
     
        
       
      
        


    
