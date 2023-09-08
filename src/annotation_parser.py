from __future__ import annotations
from shapely.geometry import Polygon
import pandas as pd
from pathlib import Path
import pathlib
import os
import numpy as np
from typing import Tuple
import json
import pdb







    
class AnnotationParser():
    """ Parses WSI Files and associated GeoJson annotations to create a composite data frame
         from both.Also returns the annotations as line items for which th eparsing didn't work 
         Requires folder containing WSI files and Annotations"""
    
    def __init__(self,image_path:pathlib.Path,labels_path:pathlib.Path)->None:
        self.image_path=image_path
        self.labels_path=labels_path
        
        
        
    
    
    
    def get_img_df(self)->pd.DataFrame:
        """use openslide to get properties,shape,levels etc.for each image"""
        img_paths=[self.image_path/fn for fn in os.listdir(self.image_path)  if (self.image_path/fn).suffix == '.tiff']
        img_df=pd.DataFrame({'image_path':img_paths})
        img_df=img_df.assign(image_name=[img_path.stem for img_path in img_paths],
                             WSI_size=[openslide.OpenSlide(img_path).dimensions for img_path in img_paths],
                      levels=[openslide.OpenSlide(img_path).level_count for img_path in img_paths],
                    downsample_levels=[{level:downsample for level,downsample in enumerate(openslide.OpenSlide(img_path).level_downsamples)} 
                           for img_path in img_paths])
        return img_df


    
    def get_coordinates_array(self,anno_row:pd.Series)->np.ndarray:
        """parse the annotation json (as a series of rows to get coordinates
           of the annotation as an array of dim n_points X 2 """
        
       
        
        try:
            geom=anno_row['geometry']
            coord_list=geom['coordinates']
            geom_type=geom['type']

             ### last 2 dimensions of every poly are n_points X 2
            # for multipolygon the list of coordinates i nested 1 level deep

             ## to get largest polygon if more than one get marked by mistake in  every annotation
            
            if geom_type=='Polygon':
                largest_poly=max([np.array(poly).reshape((np.array(poly).shape)[-2:]) for poly in coord_list],key=len)

            if geom_type=='MultiPolygon':
                largest_poly= max([max([np.array(poly).reshape((np.array(poly).shape)[-2:]) for poly in coords],key=len) 
                                   for coords in coord_list],key=len)

           
            
            return largest_poly

        
        except KeyError:
            return 'coordinates_key_error'
        
       
    
    def get_class_name_and_color(self,anno_row:pd.Series)->Tuple[str,int]:
        """Parse annotation to return class name for each anno and color assigned in QuPath"""
        
     
        
        try:
            classification=anno_row['properties']['classification']
            class_name=classification['name']
            class_color=classification['color']
            return class_name,np.array(class_color)
        
        except KeyError:
            
            
            return 'class_name_error',0
        
            
        
    
    def parse_json_file(self,json_path:pathlib.Path):
        # need to be list of dicts (anno_list)
        with open(json_path) as json_file:
            anno_list = json.load(json_file)
            if not isinstance(anno_list, list):
                anno_list=[anno_list]
            
        
        anno_df=pd.DataFrame(anno_list)
        anno_df=anno_df.assign(image_name=json_path.stem)
        #pdb.set_trace()
        return anno_df

        
    
    def get_anno_df(self,img_df:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame]:
        
        """ get annotation df including annotation for which parsing did not work (errored)"""
        
        jsons=[self.labels_path/fn for fn in os.listdir(self.labels_path)  if (self.labels_path/fn).suffix == '.geojson']
        anno_df=pd.concat([self.parse_json_file(json_path) for json_path in jsons])
        #pdb.set_trace()
        #anno_df=pd.concat([pd.read_json(json,orient='records').assign(image_name=json.stem) for json in jsons],ignore_index=True)
        
        ## enrich with image specific attrs, total size,zoom levels etc.
        anno_df=anno_df.merge(img_df,on='image_name')
        ## add coordinates as np array and shapely polygon with transormed origin 
        anno_df['coordinates']=anno_df.apply(self.get_coordinates_array,1)
        #pdb.set_trace()
        anno_df['class_name'], anno_df['colour_RGB']=zip(*anno_df.apply(self.get_class_name_and_color,1))
        
        ## select the annos with errored coordinates (due to annotation issues)
        errored=np.logical_or(anno_df['coordinates'].isin(['coordinates_key_error']),
                              anno_df['class_name']=='class_name_error')
        
        errored_df=anno_df[errored]
        anno_df=anno_df[~errored]
        
        #pdb.set_trace()
        ## use shapely to compute polygon attrs
        anno_df['polygon']=anno_df.apply(lambda x:Polygon(x['coordinates']),1)
        anno_df['area']=anno_df.apply(lambda x:x['polygon'].area,1)
        anno_df['circumference']=anno_df.apply(lambda x:x['polygon'].length,1)
        anno_df['bounds']=anno_df.apply(lambda x:np.array(x['polygon'].bounds).reshape((2,2)),1)
       
        #pdb.set_trace()
        
        return anno_df,errored_df
    
    
    
    def parse_annotations(self)->Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        """ returns anno_df,img_df and errored df in that order """ 
        img_df=self.get_img_df()
        anno_df,errored_df=self.get_anno_df(img_df)
        
        
        return anno_df,img_df,errored_df
        
