import tensorflow as tf 
import numpy as np 
import os 
from os.path import join,basename
from natsort import natsorted
from glob import glob 
import random
from random import shuffle
from PIL import Image

random.seed(101)


class bscope_frame_generator_new:

    def __init__(self,dire,data_type,resize,is_training=False):

        self.dire=dire
        self.is_training=is_training
        self.resize=resize

        self.data_type=data_type

        self.features_dires=natsorted(glob(join(self.dire,'features')+'/*'))
        self.labels_dires=natsorted(glob(join(self.dire,'labels')+'/*'))


        

    
      
    def resize_feature(self,feature,resize):

       

        feature=feature.astype(np.float32)

        feature_img=Image.fromarray(feature)
        

        resize_feature_img=feature_img.resize(resize)

        

        
        resize_feature=np.array(resize_feature_img)

        return resize_feature

        



      
    
    def preprocess_feature(self,feature):
        h,w=feature.shape

        
        feature=feature.astype(np.float16)
        #feature=(feature-feature.min())/(feature.max()-feature.min())

        feature=feature.reshape((h,w,1))

        return feature

  
        

    def __call__(self):

        if self.is_training:

            data=list(zip(self.features_dires,self.labels_dires))
            shuffle(data)
            
            self.features_dires,self.labels_dires=zip(*data)
         
           
        
        for i in range(len(self.features_dires)):

            assert basename(self.features_dires[i]) == basename(self.labels_dires[i])

            feature=np.load(self.features_dires[i])
            
            

            if self.resize:
                feature=self.resize_feature(feature,self.resize)

            feature=self.preprocess_feature(feature)
          

            label=np.load(self.labels_dires[i])
            label=label.astype(np.float16)

            if self.data_type !='mix':
                label=label[2:]

           

            yield feature,label