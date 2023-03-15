import tensorflow.compat.v1 as tf
import pandas as pd
import yaml
from argparse import ArgumentParser
import sys
import numpy as np
from sklearn import preprocessing

tf.disable_v2_behavior()

class CSVLoader():
    def __init__(self,csv_file_path=''):
        # read data
        df=pd.read_csv(csv_file_path)
        data = pd.DataFrame(df)
        
        self.inputs=np.hstack([data['number_dynamic_obstalces'], data['map_size'], data['corridor_width'], data['num_static_obstacles'], data['height'], data['robot_radius'], data['distance_variance'], data['num_obs_cv2'], data['iterations'],data['max_entropy']]).astype('float32')
        self.total_data_num=data.shape[0]

class NavPrediction():   
    def predict(args):
        #load CSV 
        data = CSVLoader(csv_file_path=args.predict_csv)
        X=pd.DataFrame(data.inputs)
        print(X)
        X=np.resize(X[0],(10,data.total_data_num)).T
        X=preprocessing.StandardScaler().fit_transform(X) 
        print(X)
        
        #load model
        model = tf.keras.models.load_model('crowdnav_model.h5')
        
        y_pred = model.predict(X)
        print(y_pred[0].T)
        y=pd.DataFrame()
        y['success_rate']=pd.DataFrame(y_pred[0])
        y['collision_rate']=pd.DataFrame(y_pred[1])
        y['timeout_rate']=pd.DataFrame(y_pred[2])
        print(y)




if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument(
        "--predict_csv",
        action="store",
        dest="predict_csv",
        default=f"dataset/crowdnav_predict.csv",
        help="path to the csv file which used for testing model performance",
        required=False,
    )
    args = parser.parse_args()
    results=NavPrediction.predict(args)
