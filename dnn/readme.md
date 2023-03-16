--
##0. Environment configuration

   follow the command:
      ```
      conda create --name tf2 python=3.7
      conda activate tf2
      pip install tensorflow==2.1.0
      conda install cudatoolkit=10.1
      conda install cudnn=7.6.5
      ```

    Also,numpy, sklearn and pandas are needed.
---
##1. navigation performance prediction
   There are 3 trained model for 3 planners(dwa, rlca and corwdnav). They can be used to predict the navigation performance. 
  
   follow the command:
   ```
   python3 ($planner)_predict.py --predict_csv="($path_to_csv).csv"
  ```
   Prediction results will be shown in terminal as dataframe.

2. train models

   follow the command:
   ```
      conda activate tf2
      python3 ($)/($planner)_train.py --train_csv="dataset/($data).csv"  --predic_csv="dataset/($data_predict).csv" 
   ```   
   After running, the predict performance will be shown in terminal and the model will be saved as a .h5 file.

   parameters:
      --train_csv  the dataset used to trainning
      --predic_csv the dataset used to test the prediction performance of trained model

      optional parameters:
      --seed         set random seed
      --verbose      if the trainning process is not needed to show, set it to 0
      --train_graph  loss graph during trainning, if you want, set it to True
      --index        if you want to change the input of the neural network, set it here, as string.
      --lr_decay     if learning rate decay is considered, set it to True

    for example:
    ```
       python3 dwa_train.py --seed=34 --train_csv="dataset/dwa.csv"  --predic_csv="dataset/dwa_predict.csv" --verbose=0 --train_graph=True --index=number_dynamic_obstalces,robot_radius,average_linear_velocity,average_obstalce_size,num_static_obstacles,corridor_width,distance_avg,distance_variance,robot_max_speed
    ```
---
##3. forward search
   selection.py is used to do the forward search to select the better subset of inputs, which optimize the performance of model.
   if you want do the forward search
   run
   ```
      python3 ($)/selection.py
   ```
   It will do the forward search on dwa_train.py, if you want to try it on other model, change the path on line 44 in script.
  

