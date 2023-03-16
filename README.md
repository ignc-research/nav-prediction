![](http://img.shields.io/badge/stability-stable-orange.svg?style=flat)
[![Linux](https://svgshare.com/i/Zhy.svg)](https://svgshare.com/i/Zhy.svg)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

# Nav-Prediction 
This repository provides tools to record navigation metrics and predict the navigation performance of planners.



- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Recording](#recording)



---

## Prerequisites
Below is the software we used. We cannot guarantee older versions of the software to work. Yet, newer software is most likely working just fine.

| Software      | Version        |
| ------------- | -------------- |
| OS            | Ubuntu 20.04.4 |
| Python        | 3.8.10         |




---
## Installation
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
Build catkin
```
cd ../../.. && catkin_make
```
For running the recording pipeline, install other requirements:
```
cd src/utils/navpred-data-recorder/pipelines/original && pip install -r requirements.txt
pip install mpi4py
```
Finish




---
## Recording
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
```
