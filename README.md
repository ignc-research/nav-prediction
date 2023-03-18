![](http://img.shields.io/badge/stability-stable-orange.svg?style=flat)
[![Linux](https://svgshare.com/i/Zhy.svg)](https://svgshare.com/i/Zhy.svg)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

# Nav-Prediction 
This repository provides the code used in our paper [Predicting Navigational Performance of Obstacle Avoidance Approaches Using Deep Neural Networks]().
It provides tools to record navigation metrics and predict the navigation performance of planners. [Link to demo video.]()


- [Training Pipeline](#training-pipeline)
- [Running the Pipeline](#running-the-pipeline)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Recording](#recording)


---
## Training Pipeline
<img src="/docs/imgs/trainning_pipeline.png">
Our pipeline consists of four main modules and neural networks:

- Map-Generator: 
This module provides variability in the input parameters and generates a different map for each simulation run.
![Map Creator](https://user-images.githubusercontent.com/73646817/226105572-fc9f0ee5-3d41-4413-bf26-a166357398bc.gif)


- [Arena-bench Simulation](https://github.com/ignc-research/arena-bench):
This module is the development platform of our previous works, which is responsible for preparing and running the simulations. It takes as input the map created by the map generated, the navigation planner and the robot to be used, and many other randomized parameters to cause variety in the simulations. The obstacles are created with randomized attributes before the first simulation run, and each preserves the same characteristics through all simulated episodes.
![start up crop](https://user-images.githubusercontent.com/73646817/226103274-48944036-7d50-4117-a002-37840caae837.gif)

- Data Recorder:
This module records the parameters that describe the simulation, and real-time data of the behavior of the robot and obstacles during all episodes of the simulation.It consists of two recorders, simulation recorder and robot recorder.
![data raw](https://user-images.githubusercontent.com/73646817/226103747-f486c05a-8f88-450d-b794-0a10ce23b3d0.gif)

- Data Transformation: 
This module conveniently create directories for each map and simulation in which all the relevant data can be found. The end result is one line in the CSV data set which represents one simulation run on a random map. The output is also stored in directories with a yaml file format, which allows the map .png file to be stored with the final data.
![training data](https://user-images.githubusercontent.com/73646817/226103949-39df156f-6b29-423c-b183-76fa553b7517.gif)

Neural Networks:
This module train the neural net works for different planners. See the detail [here](https://github.com/ignc-research/nav-prediction/tree/main/dnn).



---

## Running the Pipeline

### Prerequisites
Below is the software we used. We cannot guarantee older versions of the software to work. Yet, newer software is most likely working just fine.

| Software      | Version        |
| ------------- | -------------- |
| OS            | Ubuntu 20.04.4 |
| Python        | 3.8.10         |





### Installation
Create a catkin workspace
Clone the repo:
```
git clone git@github.com:ignc-research/nav-prediction.git
```
Change into dir:
```
cd nav-prediction
```
Ros install
```
rosws update
```
Install python pkgs, you need poetry for this
```
poetry shell&&poetry install
```
Install stable baselines
```
cd ../forks/stable-baselines3 && pip install -e .
```
Build catkin!

```
cd ../../.. && catkin_make
```
For running the recording pipeline, install other requirements:
```
cd src/utils/navpred-data-recorder/pipelines/original && pip install -r requirements.txt
pip install mpi4py
```
Finish





### Recording
Recording should running in poetry:
```
cd ($your workspace)/src/nav-prediction && poetry shell
```

To record data as .csv file, you need to go inside the dir:
```
cd ($your workspace)/src/utils/navpred-data-recorder/pipelines/original
```
Then run the command:
```
python3 pipeline_script_ver2.py —num_maps (number of maps) —num_episodes (number of episodes)
```
You can set how many maps you want to record and how many times simulation resets on a map.
For example, if you want to record 500 lines of data which based on 500 maps, and for each map, the simulation will resets 30 times, then run:
```
python3 pipeline_script_ver2.py —num_maps 500 —num_episodes 30
```
To facilitate the process of gathering only the data of recordings that finished successfully, running the command after finishing recording a batch:
```
python3 collect_records.py




