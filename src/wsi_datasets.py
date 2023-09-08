from __future__ import annotations
from torch.utils.data import Dataset,DataLoader
from shapely.geometry import Polygon,MultiPolygon
import  cv2
import numpy as np
import pandas as pd
from functools import partial
import torch
from pathlib import Path
import os
import pathlib


    


class WSI_Pyramid(Dataset):


    """Pytorch Dataset class representing a multiscale WSI dataset.inputs are img and anno dfs containing info 
         about WSI images and annotations.Pyramidal crops are sampled from the pyramid_top_levels (usually set to the most
         zoomed in level in the tiff dataset although more than one level can be used) ,with a downsample factor of one . A crop is 
         chosen from the top level of size crop_sz X crop_sz and concentric crops of the same size are chosen in the next
         num_pyramid_level levels."""
   
    
    def __init__(self,
                 anno_df:pd.DataFrame,
                 img_df:pd.DataFrame,
                 crop_pixel_size:tuple=(512,512),
                 transform=None,
                 ## accomodating Qupaths 'Other' labels
                 class2num={'Background':0,'Tumor':1,'Other':1},
                 anno_crop_prob=0.1,
                 
                 ## default set to show all levels, mostly {0:1} top level is picked
                 pyramid_top_level={0:1.0},
                 num_pyramid_levels=4, 
                 num_pyramid_mask_levels=1,
                 filter_flag=False,**kwargs)->None:
       
        
       
        self.anno_df=anno_df
        self.img_df=img_df
        ## the size in pixel of  each crop -size is kept same at 
        ## various zoom levels for batching
        self.crop_pixel_size=crop_pixel_size
        self.item_transform = transform
        self.class2num= class2num
        ## create on self.device
        ## offsets to add to crop center to get vertices
        self.offsets=np.array(self.crop_pixel_size)//2
        ## get the downsample levels common in the entire dataset
        self.common_downsample_levels=min(self.anno_df['downsample_levels'],key=len)
        ## take the intersect of user provided ds levels and the ones present in the data
        self.pyramid_top_level=pyramid_top_level
        self.pyramid_all_levels=max(self.anno_df['downsample_levels'],key=len)
        ## probability that random crop selected will be from within the annotation bound
        ## as opposed to a crop chosen from a random image at a random location
        self.anno_crop_prob=anno_crop_prob

        
        self.pyramid_top_idx=list(self.pyramid_top_level.keys())[0]
        self.pyramid_top_downsample=list(self.pyramid_top_level.values())[0]
        
        self.num_pyramid_levels=num_pyramid_levels
        self.num_pyramid_mask_levels=num_pyramid_mask_levels
        assert self.num_pyramid_levels>= self.num_pyramid_mask_levels,'num_pyramid_levels used for inputs should be more than num of target maska'
        
        ## the actual levels of the tiff pyramid used as input to the model
        self.pyramid_zoom_levels={idx:self.common_downsample_levels[idx] for idx in range(self.pyramid_top_idx,self.pyramid_top_idx+self.num_pyramid_levels)}
        
        self.filter_flag=filter_flag
        
    def __len__(self):
        return len(self.anno_df)
    
    
    
    def sample_crop_center(self,annotation_row:pd.Series):
          
        if np.random.rand()<self.anno_crop_prob:
            ## case where crop center is sampled from the annotations bounds
            
            minxy, maxxy=annotation_row['bounds']
            return np.random.randint(low=np.array(minxy)-self.offsets,high=np.array(maxxy)+self.offsets)

        else:
            ## case where random crop from all possible issue locs from the same image
            tissue_locs=self.img_df.set_index('image_name',inplace=False).loc[annotation_row['image_name']]['tissue_location']
            return  tissue_locs[np.random.randint(low=0,high=len(tissue_locs))]+self.offsets

            
                
            
            
            
           
        
    
    def get_pyramid_crops(self,annotation_row:pd.Series):
        
        """ get a random center crop at any possible zoom level from the periphery
           of an annotation """
       

       
       
        wsi_size,anno_coordinates=(np.array(x) for x in [annotation_row['WSI_size'],
                                                             annotation_row['coordinates']])

       
        random_crop_center=torch.tensor(self.sample_crop_center(annotation_row))
        # pdb.set_trace()
        offsets_arr=torch.tensor([[-1,1],[1,1],[1,-1],[-1,-1]])*self.offsets   # 4 X 2
        downsample_arr=torch.tensor(list(self.pyramid_zoom_levels.values())).unsqueeze(1).unsqueeze(1)

       
        pyramid_crops=offsets_arr.unsqueeze(0)*downsample_arr+random_crop_center.unsqueeze(0).unsqueeze(0)# Pyramid_Levels X 4 X 2  (one crop for each level)
        pyramid_crops=np.array(pyramid_crops).astype(np.int32)

        pyramid_top_lefts=pyramid_crops.min(axis=1)   # Pyramid_Levels X 2
      
       
        ## get the top left of every sampled level in the pyr
        return pyramid_crops,pyramid_top_lefts


    def filter_crops_byWSIsize(self,wsi_size:tuple,all_crops:np.ndarray):
        
         
    
        max_bounds=np.max(all_crops,axis=1).values<wsi_size.unsqueeze(0)
        min_bounds=np.min(all_crops,axis=1).values>torch.zeros_like(wsi_size.unsqueeze(0))
        
        ## get all feasible crops/tiles which are wholly within the WSI bounds, associated with that particular annotation
        all_crops=all_crops[np.logical_and(max_bounds.all(axis=1),min_bounds.all(axis=1))]

        ## if there is no possible crop that fits in the WSI image for a particular annotation and zoom level, retrn 
        ## empty tensors
        
         
        return all_crops
    
   
        
        
        
        
   
    def get_mask_per_class(self,class_annotation_data:pd.DataFrame,crop:np.ndarray,
                          downsample_factor:float)->torch.tensor:
       
        """"function to create masks of each class given the annotation data and crop(image) 
            coordinates=(4X2 shape) also the donsample factor of the crop to scale the polygon coords"""
        annotation_class=class_annotation_data['class_name'].iloc[0]
        annotation_num=self.class2num[annotation_class]
        
        
        ## select the top left point of the crop
        ## its the point with the min X and Y corrdinates (top left of image is origin)
        
        top_left=crop.min(axis=0)
        
        ## create a shapely polygon from crop to find intersections between annotations and crop
        
        crop_poly=Polygon(crop)
        
        ## create list of intersecting polygons with crop to fill with clss encoding
        
        intersects=[]
        for poly in  class_annotation_data['polygon']:
            if not crop_poly.intersects(poly):
                continue
            else:
                intersect=crop_poly.intersection(poly)
                
                if isinstance(intersect,MultiPolygon):
                    for inter in intersect.geoms:
                        ext_coords=((np.array(inter.convex_hull.exterior.coords)-top_left)//downsample_factor).astype(np.int32)
                        intersects.append(ext_coords)
                elif isinstance(intersect,Polygon):
                        ext_coords=((np.array(intersect.convex_hull.exterior.coords)-top_left)//downsample_factor).astype(np.int32)
                        intersects.append(ext_coords)
                else:
                        continue
                        
                        
                

                    
        mask=np.zeros(self.crop_pixel_size,dtype=np.uint8)
        
       
        ## fill the intersected polygons within the mask
        cv2.fillPoly(mask,intersects,color=annotation_num)
        
        return torch.tensor(mask,dtype=torch.uint8)
        
        
        
    def read_slide_region(self,slide_obj:openslide.OpenSlide,top_left:np.ndarray,
                         level:int):
        """ returns the pixel RGB from WSI given a location,crop_size and level"""
       
        return slide_obj.read_region(tuple(top_left.astype(np.int32)),level,self.crop_pixel_size)

    def get_dl(self,batch_size,kind,shuffle=True):
        ## only shuffle the train dl not the validation one
        shuffle=kind=='train'
        return DataLoader(dataset=self,batch_size=batch_size,shuffle=shuffle)

    def get_img_T(self,pyramid_top_lefts:np.ndarray,image_path:pathlib.Path):
        

        img_T=np.concatenate([np.array(self.read_slide_region(openslide.OpenSlide(image_path),sampled_top_left,zoom_level))[:,:,:-1] 
                  for zoom_level,sampled_top_left in zip(self.pyramid_zoom_levels,pyramid_top_lefts)],axis=2)
    
        img_T=torch.tensor(img_T).permute(2,0,1)
        return img_T
        
        


    def get_msk_T(self,pyramid_crops:np.ndarray,image_anno_data:pd.DataFrame):
        pyramid_msk=[]
        for i in range(self.num_pyramid_mask_levels):
            get_classwise_masks=partial(self.get_mask_per_class,
                                              crop=pyramid_crops[i],
                                              downsample_factor= list(self.pyramid_zoom_levels.values())[i])
            
            class_wise_masks=image_anno_data.groupby('class_name').apply(get_classwise_masks)
    
    
    
    
            ## stack the masks of various classes
            stacked_masks=torch.stack(class_wise_masks.to_list(),dim=0)
            
            ## create a composite mask with higher class numbers taking precedence in case of ties
            
            composite_mask=stacked_masks.max(dim=0)
            pyramid_msk.append(composite_mask.values)
        
        return torch.stack(pyramid_msk)



    
    def __getitem__(self, index):
        ## select annotation 
        anno_row=self.anno_df.iloc[index]
        
        ## select all annotations in the same image as indexed annotation
        image_name,anno_class=anno_row['image_name'],anno_row['class_name']
        dowsample_levels=anno_row['downsample_levels']
        image_path=anno_row['image_path']
        image_anno_data=self.anno_df[self.anno_df['image_name']==image_name]
        
        ## select pyramidal crops from N_levels zoom levels
        pyramid_crops,pyramid_top_lefts=self.get_pyramid_crops(anno_row)

        ## create a stack of pyramid crops centered at the annotation with as many zoom levels as descibed by pyramid_levels
        img_T=self.get_img_T(pyramid_top_lefts,image_path)
        mask_T=self.get_msk_T(pyramid_crops,image_anno_data)
    
        return img_T,mask_T




class WSI_Inference(WSI_Pyramid):


    """Pytorch Dataset class to perform inference on a WSI.Input is the tissue locations on a WSI which are obtained after
      removal of background. Inference is run on crops of 128X128 extracted from these locations.Inherits from the pyramid
      parent class to make available common convenience functions"""

    
   
    
    def __init__(self,
                 wsi_path:pathlib.Path,
                 wsi_tissue_locs:np.ndarray,
                 **kwargs)->None:
        ## init the pyramidal dataset
        super().__init__(**kwargs)

        self.wsi_path= wsi_path
        self.wsi_tissue_locs=wsi_tissue_locs

    def __len__(self):
        return len( self.wsi_tissue_locs)

    def __getitem__(self,index):
        top_left=torch.tensor(self.wsi_tissue_locs[index])
        crop_center=top_left+self.offsets
        downsample_arr=torch.tensor(list(self.pyramid_zoom_levels.values()))
        pyramid_top_lefts=crop_center-self.offsets.unsqueeze(0)* downsample_arr.unsqueeze(1)
        pyramid_top_lefts=np.array( pyramid_top_lefts).astype(np.int32)
        img_T=self.get_img_T(pyramid_top_lefts,self.wsi_path)
        ## return locations and pyramidal images for inference
        return top_left,img_T

    def get_dl(self,batch_size):
        return DataLoader(dataset=self,batch_size=batch_size,shuffle=False)
        
        

        
        

    
        
       
        
       
      
