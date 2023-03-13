import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
import math
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import torch
import os
import sys
import random
from argparse import ArgumentParser
import datetime



#############dataloader
class CSVLoader():
    def __init__(self,csv_file_path=''):
        # read data
        df=pd.read_csv(csv_file_path)
        data = np.array(df)
       
        self.robot_radius=np.hstack(data[:,1:2]).astype('float32')
        self.robot_max_speed=np.hstack(data[:,0:1]).astype('float32')
        self.width=np.hstack(data[:,7:8]).astype('float32')
        self.height=np.hstack(data[:,8:9]).astype('float32')
        self.map_type=np.hstack([data[:,9:10],data[:,12:13],data[:,15:16]]).astype('float32')
        self.iterations=np.hstack(data[:,10:11]).astype('float32')
        self.corridor_width=np.hstack(data[:,11:12]).astype('float32')
        self.num_static_obstacles=np.hstack(data[:,13:14]).astype('float32')
        self.static_obstacle_size=np.hstack(data[:,14:15]).astype('float32')
        self.mean_angle_info=np.hstack(data[:,16:17]).astype('float32')
        self.entropy=np.hstack(data[:,17:18]).astype('float32')
        self.map_size=np.hstack(data[:,18:19]).astype('float32')
        self.max_entropy=np.hstack(data[:,19:20]).astype('float32')
        self.num_obs_cv2=np.hstack(data[:,20:21]).astype('float32')
        self.occupancy_ratio=np.hstack(data[:,21:22]).astype('float32')
        self.distance_normalized=np.hstack(data[:,22:23]).astype('float32')
        self.distance_avg=np.hstack(data[:,23:24]).astype('float32')
        self.distance_variance=np.hstack(data[:,24:25]).astype('float32')     
        self.average_linear_velocity=np.hstack(data[:,25:26]).astype('float32')         
        self.average_obstalce_size=np.hstack(data[:,26:27]).astype('float32')       
        self.number_dynamic_obstalces=np.hstack(data[:,27:28]).astype('float32')
        self.planner=np.hstack(data[:,28:31]).astype('float32')
        self.map_type=np.hstack([data[:,9:10],data[:,12:13],data[:,15:16]])
        self.map_type=self.map_type.astype('float32')
        self.num_static_obstacles=np.hstack(data[:,13:14]).astype('float32').astype('float32')
        self.static_obstacle_size=np.hstack(data[:,14:15]).astype('float32').astype('float32')
        
        self.output1=np.hstack([data[:,2:3]])
        self.output2=np.hstack([data[:,3:4]])
        self.output3=np.hstack([data[:,4:5]])
        #self.input=preprocessing.StandardScaler().fit_transform(self.input.astype('float32'))
        self.output1=self.output1.astype('float32')
        self.output2=self.output2.astype('float32')
        self.output3=self.output3.astype('float32')
        self.total_data_num=data.shape[0]



class ModelTrain():
    #random seed
    def seed_tensorflow(seed):
        os.environ['PYTHONHASHSEED'] = str(seed) 
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1' 


    #learning rate decay rule
    def step_decay(num_epoch):
        initial_lrate =0.0005
        drop = 0.8
        epochs_drop = 100
        lrate = initial_lrate * math.pow(drop,math.floor((1+num_epoch)/epochs_drop))
        return lrate
    
    #add data to input set
    def input_set(index,data_loader):
        if isinstance(index,str):
            index=index.split(',')
        #print(index)
        df=pd.DataFrame()
        num_input=0
        num_column=0
        for i in index:
             num_input+=1
             if(i=='map_type'):
                num_column+=3
                ##get data
                data=pd.DataFrame()
                data=data_loader.map_type
                data=pd.DataFrame(np.resize(data[0],(data_loader.total_data_num,3)))
                # exec('data[0]=data_loader.{}'.format(i))
                
                ##add data to dataframe
                for j in range(0,3):
                    name=i+'_'+str(j)                                 
                    df[name]=data[j]
             else:
                num_column+=1
                exec('df[i]=data_loader.{}'.format(i))   
        df=preprocessing.StandardScaler().fit_transform(df)     
        return [df,num_column,num_input]
            
    ############## trainning    
    def TrainModel(args):
        
        #set random seed
        if args.seed != 0:
            ModelTrain.seed_tensorflow(args.seed)
            
        #load data       
        data_loader=CSVLoader(csv_file_path=args.train_csv)
        inputs=ModelTrain.input_set(args.index,data_loader)
        X=np.array(inputs[0])
        X_wide=[data_loader.width, data_loader.height, data_loader.occupancy_ratio]
        X_wide=pd.DataFrame(np.resize(X_wide,(data_loader.total_data_num,3)))
        X_wide=preprocessing.StandardScaler().fit_transform(X_wide)     
        # model architectrue
        input_wide=tf.keras.layers.Input(shape=3)
        input=tf.keras.layers.Input(shape=(inputs[1]))   
        drop_out = tf.keras.layers.Dropout(0.1)     
        dense1=tf.keras.layers.Dense(units=1024,kernel_regularizer=tf.keras.regularizers.l2(0.01),activation=tf.nn.relu)(input)
        #dense2=tf.keras.layers.Dense(units=32,kernel_regularizer=tf.keras.regularizers.l2(0.01),activation=tf.nn.relu)(dense1)
        #dense3=tf.keras.layers.Dense(units=8,kernel_regularizer=tf.keras.regularizers.l2(0.01),activation=tf.nn.relu)(dense2)
        merge3=tf.keras.layers.concatenate([input_wide,dense1])
        out2=tf.keras.layers.Dense(units=1,activation=tf.nn.sigmoid)(merge3)  
        
        #merge1=tf.keras.layers.concatenate([input,out2])
        #dense3=tf.keras.layers.Dense(units=1024,kernel_regularizer=tf.keras.regularizers.l2(0.01),activation=tf.nn.relu)(merge1)
        #dense5=tf.keras.layers.Dense(units=128,kernel_regularizer=tf.keras.regularizers.l2(0.01),activation=tf.nn.relu)(dense3)
        #merge4=tf.keras.layers.concatenate([input_wide,dense3])
        out1=tf.keras.layers.Dense(units=1,activation=tf.nn.sigmoid)(merge3)
        
        merge2=tf.keras.layers.concatenate([input,out1,out2])
        dense4=tf.keras.layers.Dense(units=1024,kernel_regularizer=tf.keras.regularizers.l2(0.01),activation=tf.nn.relu)(merge2)
        #dense5=tf.keras.layers.Dense(units=8,kernel_regularizer=tf.keras.regularizers.l2(0.01),activation=tf.nn.relu)(drop_out(dense4))
        #dense6=tf.keras.layers.Dense(units=8,kernel_regularizer=tf.keras.regularizers.l2(0.01),activation=tf.nn.relu)(drop_out(dense5))
        
        merge5=tf.keras.layers.concatenate([input_wide,drop_out(dense4)])    
        out3=tf.keras.layers.Dense(units=1,activation=tf.nn.sigmoid)(merge5)
        
       
        
        model=tf.keras.Model([input_wide,input],[out1,out2,out3])
        
        #trainning parameters 
        num_epochs = 1000
        batch_size=128
        lr=0.0001
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
        loss=tf.keras.losses.mae
             
        
        ########trainning model
        model.compile(optimizer=optimizer,loss=[loss,loss,loss], metrics=['mse'])
        if args.lr_decay=='True':
            print("learning decay")
            lrate = tf.keras.callbacks.LearningRateScheduler(ModelTrain.step_decay)
            history=model.fit([X_wide,X], [data_loader.output1,data_loader.output2,data_loader.output3],epochs=num_epochs,batch_size=batch_size,callbacks=[lrate],verbose=args.verbose,validation_split=0.33,)
        else:
            history=model.fit([X_wide,X], [data_loader.output1,data_loader.output2,data_loader.output3],epochs=num_epochs,batch_size=batch_size,verbose=args.verbose,validation_split=0.33,)
        if args.selection==False:
            model.summary()


        #tranning gragh
        if args.train_graph=='True':
            plt.figure()
            plt.plot(history.history['loss'],label='training loss')
            plt.plot(history.history['val_loss'],label='val loss')
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(loc='lower right')
            plt.show()

        #########use trained model to predict new data       
        test_data=CSVLoader(csv_file_path=args.predic_csv)
        x=ModelTrain.input_set(args.index,test_data)
        x_wide=[test_data.width, test_data.height, test_data.occupancy_ratio]
        x_wide=pd.DataFrame(np.resize(x_wide,(test_data.total_data_num,3)))
        x_wide=preprocessing.StandardScaler().fit_transform(x_wide)
        prediction=model.predict([x_wide,np.array(x[0])])
        if args.selection==False:
            print(prediction)
        y1=test_data.output1
        y2=test_data.output2
        y3=test_data.output3
        R2_s=r2_score(y1,prediction[0])
        R2_c=r2_score(y2,prediction[1])
        R2_t=r2_score(y3,prediction[2])
        if args.selection==False:
            print("R2_s=",R2_s)
            print("R2_c=",R2_c)
            print("R2_t=",R2_t)
        n=test_data.total_data_num
        Adjusted_R2_s=1-((1-R2_s)*(n-1))/(n-x[2]-1)
        Adjusted_R2_c=1-((1-R2_c)*(n-1))/(n-x[2]-1)
        Adjusted_R2_t=1-((1-R2_t)*(n-1))/(n-x[2]-1)
        if args.selection==False:
            print("Adjusted_R2_s=",Adjusted_R2_s)
            print("Adjusted_R2_c=",Adjusted_R2_c)
            print("Adjusted_R2_t=",Adjusted_R2_t)
        R2=(R2_s+R2_c+R2_t)/3
        Adjusted_R2=(Adjusted_R2_s+Adjusted_R2_c+Adjusted_R2_t)/3
        MSE_s=mean_squared_error(y1,prediction[0])
        RMSE_s=MSE_s**0.5
        MSE_c=mean_squared_error(y2,prediction[1])
        RMSE_c=MSE_c**0.5
        MSE_t=mean_squared_error(y3,prediction[2])
        RMSE_t=MSE_t**0.5
        MSE_avg=(MSE_s+MSE_c+MSE_t)/3
        RMSE_avg=MSE_avg**0.5
        MAE_s=mean_absolute_error(y1,prediction[0])
        MAE_c=mean_absolute_error(y2,prediction[1])
        MAE_t=mean_absolute_error(y3,prediction[2])
        MAE_avg=(MAE_s+MAE_c+MAE_t)/3
        MAPE_s=np.mean(np.abs((prediction[0]-y1)/y1))*100
        MAPE_c=np.mean(np.abs((prediction[1]-y2)/y2))*100
        MAPE_t=np.mean(np.abs((prediction[2]-y3)/(y3)))*100
        
        SSE_s = np.sum((y1 - prediction[0]) ** 2)
        SSE_c = np.sum((y2 - prediction[1]) ** 2)
        SSE_t = np.sum((y3 - prediction[2]) ** 2)
        SSE_avg=(SSE_s+SSE_c+SSE_t)/3
        if args.selection==False:
            print("R2=", R2)
            print("Adjusted_R2=",Adjusted_R2)
            print("MSE=",MSE_avg)
            print("RMSE=",RMSE_avg)
            print("MAE=",MAE_avg)
            print("MAPE_s=",MAPE_s,"%")
            print("MAPE_c=",MAPE_c,"%")
            print("SSE=", SSE_avg)
        
        
        #########error graph
        #if args.error_graph==True:
            #error graph of 1st output
            plt.figure()
            plt.plot(prediction[0],label='predicted value')
            plt.plot(y1,label='actual value')
            plt.title('error')
            plt.ylabel('value')
            plt.xlabel('')
            plt.legend(loc='lower right')
            plt.show()
            #error graph of 2nd output
            plt.figure()
            plt.plot(prediction[1],label='predicted value')
            plt.plot(y2,label='actual value')
            plt.title('error')
            plt.ylabel('value')
            plt.xlabel('')
            plt.legend(loc='lower right')
            plt.show()
            #error graph of 3rd output
            plt.figure()
            plt.plot(prediction[2],label='predicted value')
            plt.plot(y3,label='actual value')
            plt.title('error')
            plt.ylabel('value')
            plt.xlabel('')
            plt.legend(loc='lower right')
            plt.show()
         
        return [R2,Adjusted_R2]

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        dest="seed",
        default = 0,
        help="random seed"
    )
    parser.add_argument(
        "--lr_decay",
        dest="lr_decay",
        default=False,
        help="if the learning rate decay is considered, set it as Ture"
    )
    parser.add_argument(
        "--train_csv",
        action="store",
        dest="train_csv",
        default=f"/home/parallels/dnn/new/2002rlca.csv",
        help="path to the csv file which used for training neural network",
        required=False,
    )
    parser.add_argument(
        "--predic_csv",
        action="store",
        dest="predic_csv",
        default=f"/home/parallels/dnn/new/2002rlca_predict.csv",
        help="path to the csv file which used for testing model performance",
        required=False,
    )
    parser.add_argument(
        "--verbose",
        type=int,
        dest="verbose",
        default=0,
        help="choose how the tranning process be presented"
    )
    parser.add_argument(
        "--train_graph",
        dest="train_graph",
        default=False,
        help="if you want to generate a tranning loss graph,set it as True"
    )
    parser.add_argument(
        "--error_graph",
        dest="error_graph",
        default=True,
        help="if you want to generate a comparison graph between predected value and actual value,set it as True"
    ) 
    parser.add_argument(
        "--index",
        dest="index",           
        default=['robot_max_speed', 'robot_radius', 'width', 'height', 'iterations', 'corridor_width',  'mean_angle_info', 'entropy', 'map_size', 'max_entropy', 'num_obs_cv2', 'occupancy_ratio', 'distance_normalized', 'distance_avg','distance_variance','average_linear_velocity','average_obstalce_size','number_dynamic_obstalces', 'map_type','num_static_obstacles','static_obstacle_size']
       
    )  
    parser.add_argument(
        "--selection",
        dest="selection",
        default=False,
    )    
    
    args = parser.parse_args()
    results=ModelTrain.TrainModel(args)
    print(results)
