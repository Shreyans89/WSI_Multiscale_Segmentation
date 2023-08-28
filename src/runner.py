from __future__ import annotations
from sklearn.model_selection import train_test_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar,EarlyStopping

from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm






class experiment_runner():
    
    
    def __init__(self,root=Path('training_data')
                 
                 
               ):
            
            self.label_path=root/'labels'
            self.image_path=root/'images'
            
            
            self.inference_path=Path('inference')

            self.inference_image_path=self.inference_path/'images'
            ## path to store predictions (as json files)
            self.inference_label_path=self.inference_path/'labels'
            
            self.parser=AnnotationParser(self.image_path,self.label_path)
            self.anno_df,self.img_df,self.errored=self.parser.parse_annotations()
            ## precompute tissue locations for training images
            
            t_locs=self.img_df['image_path'].apply(self.preprocess,1)
            self.img_df=self.img_df.assign(tissue_location=t_locs)
            self.img_df['inference_len']=self.img_df['tissue_location'].apply(len,1)
            ## device to perform inference on
            self.device='cuda' if torch.cuda.is_available() else 'cpu'
        
            
            
          
            
    def preprocess(self,img_path:pathlib.Path,downsample_factor=512,gray_background=np.array([236, 236, 236]),tol=0)->np.ndarray:
        """ preprocess WSI to remove grey areas returns pixel locations
            from the downsampled (thumbnail) image where the tissue exists
              """
        
        
        slide=openslide.OpenSlide(img_path)
        H,W=slide.dimensions
       
        thumbnail_img=np.array(slide.get_thumbnail((H//downsample_factor,W//downsample_factor)))
        h,w,c=thumbnail_img.shape
        
        rgb_upper_bound=gray_background+tol
        rgb_lower_bound=gray_background-tol
        grey_mask = cv2.inRange( thumbnail_img, rgb_lower_bound,  rgb_upper_bound)
        
        
        num_comps, labelled_mask = cv2.connectedComponents(~grey_mask)

        tissue_comps=[]
        tissue_mask=[]
        
        for i in range(1,num_comps):
            comp_mask=(labelled_mask==i)
            #pdb.set_trace()
            unique_rgb_vals=np.unique(thumbnail_img.reshape((h*w,c))[comp_mask.flatten()],axis=0)
            if len(unique_rgb_vals)>1:
               tissue_comps.append(np.argwhere(comp_mask))
               tissue_mask.append(comp_mask)

     
                    
                    
            
        return np.concatenate(tissue_comps)*downsample_factor
        
        

   
    
    
    
    def get_dls(self,
               batch_size=32,
                num_pyramid_levels=4,
                num_pyramid_mask_levels=1,
                crop_pixel_size=(128,128),
                pyramid_top_level={0: 1.0}
                
                ):

            #filtering out image_name with less than 2 annotations for stratifying using train_test_split
            vc=self.anno_df['image_name'].value_counts()
            filt_df=self.anno_df.merge(vc.reset_index())
            ## select images with more than the median number of annotations
            filt_df=filt_df[filt_df['count']>50]
            #filt_df=filt_df[filt_df['count']==max(vc)]
           
            df_train,df_val=train_test_split(filt_df,test_size=0.2,random_state=42,stratify=filt_df['image_name'])
            ds_train,ds_val=(WSI_Pyramid(anno_df=df,
                                         crop_pixel_size=crop_pixel_size,
                                         pyramid_top_level=pyramid_top_level,
                                         num_pyramid_levels=num_pyramid_levels,
                                         num_pyramid_mask_levels=num_pyramid_mask_levels) for df in (df_train,df_val))
            
           
            dl_train,dl_val=(dset.get_dl(batch_size=batch_size,kind=kind) for dset,kind in zip((ds_train,ds_val),('train','val')))
            
        
            return dl_train,dl_val


    def get_inference_dl(self,
                         wsi_img_name,
                        batch_size=32,
                         
                         **kwargs):
        
        wsi_path=self.inference_image_path/wsi_img_name
        tissue_locs=self.preprocess(wsi_path)

        ## pass extra args on num pyramid levels etc. to the dataset class
        ds_inference=WSI_Inference(wsi_path= wsi_path,wsi_tissue_locs=tissue_locs,**kwargs)
        dl_inference= ds_inference.get_dl( batch_size=batch_size)
        return dl_inference
    
    
    def get_inference_schema(self,sample_json='inference_schema.geojson'):
        json_path=self.inference_path/sample_json
        with open(json_path) as json_file:
             schema = json.load(json_file)

        return schema
        
    
    
    
    
    def run_inference(self,checkpoint_path:pathlib.Path,wsi_img_name:str,batch_size=32,max_iter=10,**kwargs):
        inference_dl=self.get_inference_dl(wsi_img_name=wsi_img_name,batch_size=batch_size,anno_df=self.anno_df,**kwargs)
       
        ## setting pl_module to eval mode
        inference_model=SegLightningModule.load_from_checkpoint(checkpoint_path)
        inference_model.eval()

        inference_filename='.'.join((wsi_img_name,'geojson'))
        inference_filepath=self.inference_label_path/inference_filename
        
        inference_json=[]
        with torch.no_grad():
          for i,(top_left_b,img_b) in enumerate(tqdm(inference_dl)):
               top_left_b,img_b=top_left_b.to(self.device),img_b.to(self.device)
               if i>max_iter:
                   break
               pred_b= inference_model(img_b.to(torch.float32))
               pred_b=torch.argmax(pred_b,dim=1)
               pred_coords=torch.cat([torch.argwhere(pred)+top_left for pred,top_left in zip(pred_b,top_left_b) if pred.sum()>0])
               coords_list=pred_coords.tolist()
               if len(coords_list)>0:
                  inference_schema=self.get_inference_schema()
                  inference_schema['geometry'][ 'coordinates']=coords_list
                  inference_json.append(inference_schema)
                  ## write in append mode
                  with open(inference_filepath, 'w') as json_file:
                     json.dump(inference_json, json_file, indent=4)

        
       
        
      
                   
                   
                   
                
        
    
    
    def show_batch(self,kind='train',
                   mask_color=torch.tensor((0,128,255)),
                   num_pyramid_levels=4,
                   crop_pixel_size=(512,512),
                   pyramid_top_level={0: 1.0},
                   show_batch_size=8,save=False,**kwargs,
                   ):
                   
              

          dl_train,dl_val=self.get_dls(batch_size=show_batch_size,
                                                   num_pyramid_levels=num_pyramid_levels,
                                                   num_pyramid_mask_levels=num_pyramid_levels,
                                                   crop_pixel_size=crop_pixel_size,
                                                   pyramid_top_level=pyramid_top_level
                                                   )



            
            
          if kind=='train':
            
            img_b,mask_b=next(iter(dl_train))
            #pdb.set_trace()

          if kind=='val':
            
             img_b,mask_b=next(iter(dl_val))

        
          H,W=crop_pixel_size
          img_b=img_b.reshape(batch_size*num_pyramid_levels,3,H,W)

          #pdb.set_trace()
          mask_b=mask_b.reshape(batch_size*num_pyramid_levels,1,H,W)

        
          fig = plt.figure(figsize=(32,24))
          color_b=mask_b*mask_color.unsqueeze(0).unsqueeze(2).unsqueeze(3)
          overlay_b=img_b+color_b
          grid_img=make_grid(overlay_b,nrow=num_pyramid_levels)
          
          if save:
            plt.imshow(grid_img.permute(1,2,0))
              
          plt.imshow(grid_img.permute(1,2,0))
         

         
    
       
    
    def run_training(self,name,epochs=10,
                     num_pyramid_levels=4,
                     num_pyramid_mask_levels=1,
                     crop_pixel_size=(512,512),
                     pyramid_top_level={0: 1.0},
                     batch_size=32,
                     **kwargs):

        dl_train,dl_val=self.get_dls(batch_size=batch_size,
                                     num_pyramid_levels=num_pyramid_levels,
                                     num_pyramid_mask_levels=num_pyramid_mask_levels,
                                     crop_pixel_size=crop_pixel_size,
                                     pyramid_top_level=pyramid_top_level)

        self.pl_module=SegLightningModule(in_channels=3*num_pyramid_levels,**kwargs)
                                                   

        
        
   
        accelerator='gpu' if torch.cuda.is_available() else 'cpu'
        
        self.trainer = pl.Trainer(max_epochs=epochs, accelerator=accelerator,callbacks=[EarlyStopping(monitor='Val_loss',
                                                             patience=5,  verbose=True, mode='min' ),
                                       TQDMProgressBar(refresh_rate=1)],
                                logger=CSVLogger(flush_logs_every_n_steps=10,save_dir='runs',name=name))
                                                                         
                                                        
                                                              
                                               
        
        ## automatically saves model
        self.trainer.fit(model=self.pl_module, train_dataloaders=dl_train,val_dataloaders=dl_val)
            

            
            
