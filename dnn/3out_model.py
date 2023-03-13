import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
import math
from sklearn.metrics import r2_score
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
        ###input after selection
        #self.input=np.hstack([data[:,6:7],data[:,75:76],data[:,36:51]])
        ###all input
        #self.input=np.hstack([data[:,3:7],data[:,21:66],data[:,70:79]])
        #self.input=np.hstack([data[:,5:6],data[:,75:76],data[:,6:7],data[:,72:73],data[:,36:51]])
        self.robot_radius=np.hstack(data[:,3:4]).astype('float32')
        self.robot_max_speed=np.hstack(data[:,4:5]).astype('float32')
        self.number_dynamic_obs=np.hstack(data[:,5:6]).astype('float32')
        self.number_static_obs=np.hstack(data[:,6:7]).astype('float32')
        self.num_vertices_static_obs=np.hstack(data[:,21:36]).astype('float32')
        self.size_dynamic_obs=np.hstack(data[:,36:51]).astype('float32')
        self.speed_dynamic_obs=np.hstack(data[:,51:66]).astype('float32')
        self.AngleInfo=np.hstack(data[:,70:71]).astype('float32')
        self.Entropy=np.hstack(data[:,71:72]).astype('float32')
        self.MapSize=np.hstack(data[:,72:73]).astype('float32')
        self.MaxEntropy=np.hstack(data[:,73:74]).astype('float32')
        self.NumObs_Cv2=np.hstack(data[:,74:75]).astype('float32')
        self.OccupancyRatio=np.hstack(data[:,75:76]).astype('float32')
        self.distance_normalized=np.hstack(data[:,76:77]).astype('float32')
        self.distance_avg=np.hstack(data[:,77:78]).astype('float32')
        self.distance_variance=np.hstack(data[:,78:79]).astype('float32')
        self.output1=np.hstack([data[:,1:2]])
        self.output2=np.hstack([data[:,2:3]])
        self.output3=np.hstack([data[:,66:67]])
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
    def step_decay(num_epochs):
        initial_lrate =lr
        drop = 0.2
        epochs_drop = 1000.0
        lrate = initial_lrate * math.pow(drop,math.floor((1+num_epochs)/epochs_drop))
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
             if(i=='speed_dynamic_obs' or i=='num_vertices_static_obs' or i=='size_dynamic_obs'):
                num_column+=15
                ##get data
                data=pd.DataFrame()
                exec('data[0]=data_loader.{}'.format(i))
                data=pd.DataFrame(np.resize(data[0],(data_loader.total_data_num,15)))
                ##add data to dataframe
                for j in range(0,15):
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
        #print(X)
        
        # model architectrue
        input=tf.keras.layers.Input(shape=(inputs[1]))   
        drop_out = tf.keras.layers.Dropout(0.1)     
        dense1=tf.keras.layers.Dense(units=256,activation=tf.nn.relu)(drop_out(input))
        out1=tf.keras.layers.Dense(units=1,activation=tf.nn.sigmoid)(dense1)
        out2=tf.keras.layers.Dense(units=1,activation=tf.nn.sigmoid)(dense1)     
        out3=tf.keras.layers.Dense(units=1,activation=tf.nn.sigmoid)(dense1)
        
        model=tf.keras.Model(input,[out1,out2,out3])
        
        #trainning parameters 
        num_epochs = 500
        batch_size=128
        lr=0.01
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
        loss=tf.keras.losses.mae
             
        
        ########trainning model
        model.compile(optimizer=optimizer,loss=[loss,loss,loss], metrics=['mse'])
        if args.lr_decay==True:
            lrate = tf.keras.callbacks.LearningRateScheduler(step_decay(num_epochs))
            history=model.fit(X, [data_loader.output1,data_loader.output2,data_loader.output3],epochs=num_epochs,batch_size=batch_size,callbacks=[lrate],verbose=args.verbose,validation_split=0.33,)
        else:
            history=model.fit(X, [data_loader.output1,data_loader.output2,data_loader.output3],epochs=num_epochs,batch_size=batch_size,verbose=args.verbose,validation_split=0.33,)
        if args.selection==False:
            model.summary()


        #tranning gragh
        if args.train_graph==True:
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
        prediction=model.predict(np.array(x[0]))
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
        if args.selection==False:
            print("R2=", R2)
            print("Adjusted_R2=",Adjusted_R2)
        return [R2,Adjusted_R2]
        #########error graph
        if args.error_graph==True:
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
        default=f"/home/parallels/dnn/data1218.csv",
        help="path to the csv file which used for training neural network",
        required=False,
    )
    parser.add_argument(
        "--predic_csv",
        action="store",
        dest="predic_csv",
        default=f"/home/parallels/dnn/test1217.csv",
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
        default=False,
        help="if you want to generate a comparison graph between predected value and actual value,set it as True"
    ) 
    parser.add_argument(
        "--index",
        dest="index",           default=['robot_radius','robot_max_speed','number_dynamic_obs','number_static_obs','num_vertices_static_obs','size_dynamic_obs','speed_dynamic_obs','AngleInfo','Entropy', 'MapSize','MaxEntropy','NumObs_Cv2','OccupancyRatio','distance_normalized','distance_avg','distance_variance']
    )  
    parser.add_argument(
        "--selection",
        dest="selection",
        default=False,
    )    
    args = parser.parse_args()
    results=ModelTrain.TrainModel(args)
    print(results)
