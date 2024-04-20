import tensorflow as tf
import numpy as np
import os
from os.path import exists,basename
from time import time
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image



class Inference:
    
    def get_feature(self,feature_dire,reshaped_size): ## we will provide RADAR image dire saved in numpy file

        

        
        feature=self.preprocess_feature(feature_dire,reshaped_size) ## Feature are preprocessed with shape of : batch_size,h,w,c==>1,512,512,1

        return feature
    



    def get_label(self,feature_dire): ## we will provide RADAR image dire saved in numpy file

        

        details=basename(feature_dire).split('.')[0].split('_')[:-3] ## We take the labels decoded in the filenaming system
  
        
        t1,t2,dt1,dt2=details[1:]
        
        label=np.array([t1,t2,dt1,dt2],dtype=np.float16)
        

        return label
    



    def resize_feature(self,feature,resize):

       

        feature=feature.astype(np.float32)

        feature_img=Image.fromarray(feature)
        

        resize_feature_img=feature_img.resize(resize)

        

        
        resize_feature=np.array(resize_feature_img)

        return resize_feature

        

    
    def preprocess_feature(self,feature_dire,reshaped_size): ## To make the feature ready to be processed by loaded model

        feature=np.load(feature_dire)
        resize=(reshaped_size,reshaped_size)
        feature=self.resize_feature(feature,resize=resize)
        h,w=feature.shape
        feature=feature.astype(np.float16)
        feature=feature.reshape((-1,h,w,1))
        return feature

    
    def __call__(self,feature_dire,reshaped_size,model):

        feature=self.get_feature(feature_dire,reshaped_size) ## return feature and true label

        start_time=time()
      
        predicted_label=model.predict(feature) ## Predicted Label

        infer_time=time()-start_time

        return predicted_label[0],infer_time # Return true_label, and predicted_label
