import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
import math
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import torch
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split, RandomizedSearchCV 
from sklearn.linear_model import LogisticRegression 
from sklearn import metrics 
from sklearn.preprocessing import MinMaxScaler
import os
import sys


all_inputs=['robot_radius', 'robot_max_speed','number_dynamic_obs', 'number_static_obs',  'num_vertices_static_obs', 'size_dynamic_obs','speed_dynamic_obs', 'AngleInfo', 'Entropy', 'MapSize', 'MaxEntropy', 'NumObs_Cv2', 'OccupancyRatio', 'distance_normalized', 'distance_avg', 'distance_variance']

class selection():
  
  def iteration():
     global all_inputs
     R2 = 0
     Adjusted_R2 = 0
     df=pd.DataFrame()
     index=np.array(list(df))
     state=0
     index_temp=''
     selecting_index=np.array(list(df))
     while(len(all_inputs)>0):
        selecting_index=index
        for i in all_inputs:  
            string=''   
            selecting_index=np.append(selecting_index,i)
            print("selecting_index:",selecting_index)
            for j in range(0,len(selecting_index)):
               string+=selecting_index[j]
               if j!=len(selecting_index)-1:
                   string+=','
            print(string)   
            command='python3 dnn/3out_model.py --seed=34 --train_csv="/home/parallels/dnn/data1218.csv" --predic_csv="/home/parallels/dnn/test1217.csv" --selection=True --verbose=0 --index={}'.format(string)
            temp=os.popen(command).read()
            temp=temp.replace('[','')
            temp=temp.replace(']','')
            temp=temp.split(',')
            print("result",temp)
            R2_temp=float(temp[0])
            Adjusted_R2_temp=float(temp[1])
            print("R2=",R2_temp)
            print("Adjusted_R2=",Adjusted_R2_temp)
            selecting_index=np.delete(selecting_index, len(selecting_index)-1)
            if(R2_temp>R2 and Adjusted_R2_temp>Adjusted_R2):
                R2=R2_temp
                Adjusted_R2=Adjusted_R2_temp
                index_temp=i
                state=1                                    
            print("index_temp=",index_temp)                        
            print("----------------------------------------")
            print("----------------------------------------")
        if(state==0):
            break  
        index=np.append(index,index_temp)
        all_inputs.remove(index_temp)
        state=0   
     return [index,R2,Adjusted_R2]

results=selection.iteration()    
print("selected inputs:" ,results[0])   
print("R2:", results[1])
print("Adjusted_R2:",results[2])
