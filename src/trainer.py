import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from src.cnn_model import * ## Import cnn model

from utils.generator import bscope_frame_generator_new

import os
from os.path import join,basename,exists
from utils.utils import write_pickle

import optuna
from optuna.trial import TrialState

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from natsort import natsorted



class Trainer:

    def __init__(self,logger,config):
        ## Information of training dataset
        """
        Here, we will initialise the configuration from config file. 
        """
        self.data_type=config['data_type']
        self.data_dire=config[self.data_type]['dire']
        self.num_labels=config[self.data_type]['num_labels']

        self.train_dire=join(self.data_dire,'train')
        
        self.val_dire=join(self.data_dire,'valid')
        self.test_dire=join(self.data_dire,'test')

        print('tesing: ',os.listdir(self.test_dire))
        
        
     
        
        
        self.opt=config['opt']
        #General Information about budilding model
        self.input_shape=(config['input_shape'],config['input_shape'],1)
        self.batch_size=config['BATCH_SIZE']
        self.num_epochs=config['EPOCHS']
        self.learning_rate= config['learning_rate']


           ## Information of size of CNN model

        self.model_size=config['model_size']
        self.save_dire=join(config['save_dire'],str(self.input_shape[0]),self.data_type,self.model_size)

        if not os.path.exists(self.save_dire):
            os.makedirs(self.save_dire)


     
       
    
        self.logger=logger ## Logger is used to write the training information in text file
    

 
    
    def build_model(self): # Build the model based on selected version of CNN


     
        if self.model_size=='small':

            self.model=cnn_small(input_shape=self.input_shape,num_classes=self.num_labels)
            print(self.model.summary())
        
        elif self.model_size=='medium':

            self.model=cnn_medium(input_shape=self.input_shape,num_classes=self.num_labels)

        if self.model_size=='large':

            self.model=cnn_large(input_shape=self.input_shape,num_classes=self.num_labels)

        
        if self.opt=='adam':
        

            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        elif self.opt=='sgd':
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate)

        self.model.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['mse'])

        

        self.checkpoint=tf.keras.callbacks.ModelCheckpoint(filepath=self.save_dire,
                                                               monitor="val_loss",
                                                               save_best_only=True,
                                                               mode='min',
                                                               save_weights_only=False,
                                                        )
        
    def get_data(self):
        """
        Building tf training_pipeline Here we used tf.data.generator This is because in the future if you would like to train on higher feature size then your RAM may not be enough. Thus with 
              genertor you do not need to worry about it     """

        output_signature = (tf.TensorSpec(shape = self.input_shape, dtype = tf.float16),
                    tf.TensorSpec(shape = self.num_labels, dtype = tf.float16))

        
        cache_train=join('cache',str(self.input_shape[0]),self.data_type,'train/') ## path system of cache file
        cache_test=join('cache',str(self.input_shape[0]),self.data_type,'test/')
        cache_val=join('cache',str(self.input_shape[0]),self.data_type,'valid/')

        if not exists(cache_train): ## Create the cache folder if not existed
            os.makedirs(cache_train)

        if not exists(cache_test):
            os.makedirs(cache_test)

        if not exists(cache_val):
            os.makedirs(cache_val)

        self.train_ds=tf.data.Dataset.from_generator(bscope_frame_generator_new(dire=self.train_dire,data_type=self.data_type,resize=self.input_shape[:2],
                                                                               is_training=True),output_signature=output_signature)



        
        self.train_ds=self.train_ds.cache(cache_train)
        self.train_ds=self.train_ds.batch(self.batch_size)
        

        self.train_ds=self.train_ds.prefetch(tf.data.AUTOTUNE)


        self.val_ds=tf.data.Dataset.from_generator(bscope_frame_generator_new(dire=self.val_dire,data_type=self.data_type,resize=self.input_shape[:2],
                                                                               is_training=True),output_signature=output_signature)


        
        self.val_ds=self.val_ds.cache(cache_val)
        self.val_ds=self.val_ds.batch(batch_size=self.batch_size)
        

        self.val_ds=self.val_ds.prefetch(tf.data.AUTOTUNE)

        self.test_ds=tf.data.Dataset.from_generator(bscope_frame_generator_new(dire=self.test_dire,data_type=self.data_type,
                                                                               resize=self.input_shape[:2],
                                                                               is_training=False),output_signature=output_signature)

        
        self.test_ds=self.test_ds.cache(cache_test)
        self.test_ds=self.test_ds.batch(batch_size=self.batch_size)
        
        self.test_ds=self.test_ds.prefetch(tf.data.AUTOTUNE)

      

    

    def do_training(self):
        self.logger.save_log('Data Type: {}; Selected Model: {}'.format(self.data_type,self.model_size))

        self.get_data()
       
        self.build_model()

        history=self.model.fit(self.train_ds,batch_size=self.batch_size,epochs=self.num_epochs,
                               validation_data=self.val_ds,verbose=True,callbacks=[self.checkpoint])

        history_save_folder='history_logs/{}/{}/{}/'.format(self.input_shape,self.data_type,self.model_size)

        if not exists(history_save_folder):
            os.makedirs(history_save_folder)

        
            
        history_save_path=history_save_folder+'/history.pickle' ## Save the history of training in a pickle file

        write_pickle(history_save_path,history.history)

        tf.keras.backend.clear_session()

        
        
        loaded_model=tf.keras.models.load_model(self.save_dire)

        self.model.set_weights(loaded_model.get_weights())

        y_pred=self.model.predict(self.test_ds)

        y_true_ds=self.test_ds.map(lambda x,y:y)

        y_true=np.array(list(y_true_ds.unbatch().as_numpy_iterator()))
        overall_mse=mean_squared_error(y_true, y_pred)
        thres_mse=mean_squared_error(y_true[:,-2:], y_pred[:,-2:])
        print()
        self.logger.save_log('\nOverall Loss :{}'.format(overall_mse))
        self.logger.save_log('\nthres_loss :{}'.format(thres_mse))

        return thres_mse ## return the mse lose

        # if needed save history from here

       
    
    

      

       
        






        



        

        






       
