# Baxter MPNet Experiments

MPNet algorithm implemented and tested for use with the Baxter Research Robot in a set of set of realistic obstacle scenes for motion planning experiments.  


# ROS, Baxter, and MoveIt! System Dependencies
Install ROS and the necessary ROS packages below in a catkin workspace.

[ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu)

[Baxter SDK](http://sdk.rethinkrobotics.com/wiki/Workstation_Setup)

[Baxter Simulator](http://sdk.rethinkrobotics.com/wiki/Simulator_Installation)

[MoveIt!](https://moveit.ros.org/install/)

[Baxter MoveIt! Configuration](http://sdk.rethinkrobotics.com/wiki/MoveIt_Tutorial)

### Virtual Environment Setup
Navigate to wherever you keep your Python virtual environments, and create a new one for PyTorch with Python 2.7, and the other Python packages required to run (replace $PYTHON2_PATH with the absolute path to your system's python 2.7 -- for example, ```/usr/bin/python```). See this [link](https://help.dreamhost.com/hc/en-us/articles/215489338-Installing-and-using-virtualenv-with-Python-2) for more details on Python virtual environments.

```
virtualenv pytorch-python2 -p $PYTHON2_PATH
```
and then install the dependencies from the requirements.txt file found in the root of this repository, after activating the virtual environment
```
source /path/to/pytorch-python2/bin/activate
pip install -r requirements.txt
```

# Install and Download Sample Data
In a catkin workspace, clone repo within the source folder and build the workspace 
```
cd /path/to/catkin_workspace/src/
git clone https://github.com/anthonysimeonov/baxter_mpnet_experiments.git
catkin build
```

Navigate to ```data``` folder, download and unzip [this](https://drive.google.com/file/d/1WMK_uoKzAuetUXcO_suJc2meG9zQrPrq/view?usp=sharing) file to obtain a sample dataset for paths, collision free targets, and point clouds. Navigate to ```models``` folder, and download and unzip [this](https://drive.google.com/file/d/1iblAH9u5xZsR1_222IgHaZExC8l1sX1P/view?usp=sharing) file to obtain a set of trained neural network models that produce similar results as described in the paper.

# MPNet Pipeline

### Dataset Generation
Configure whatever expert planner you want to use by modifying the ```ompl_planning.yaml``` file in the ```baxter_moveit_config``` package (found under the ```config``` directory, ```RRTStarkConfig``` works well). Rebuild workspace if any changes here have been made, then launch environment.

```
roslaunch baxter_mpnet_experiments baxter-mpnet.launch
```

and then run training data generation python

```
python path_data_generation.py
```

### Model Training
Make sure virtual environment is sourced
```
source /path/to/environments/pytorch-python2/bin/activate
```
and then run training (make sure ```run_training.sh``` has been made executable)

```
./run_training.sh
```

### Model Testing
Make sure that both the Baxter planning scene are launched and PyTorch Python2 virtual environment are sourced, and then run the test script
```
./run_testing.sh
```

### Data Analysis and Path Playback
During testing, the MPNet algorithm will plan paths and a collision checker will verify if they are feasible or not. The paths are then stored in the local ```path_samples``` directory, and can be played back on the simulated Baxter with the ```path_playback_smooth.py``` script. This is a very minimal working example of playing a saved path from start to goal, so USE WITH CAUTION --- ESPECIALLY IF RUNNING ON THE REAL ROBOT.

```
python path_playback_smooth.py
```

# Docker container with system dependencies
The experiments can alternatively be run in a container which has all the system dependencies set up, if the local system is incompatible with any of the supporting packages/libraries. The docker requires the local system to have a GPU and Nvidida drivers compatible with CUDA 9.0 (it can be easily adapted to work with CPU-only systems). To build the container, first download and unzip [this](https://drive.google.com/file/d/1gSWmqudfR9_tL6QGkVxBu8yuyjzt0w46/view?usp=sharing) folder in the ```docker``` directory, which contains some of the resources necessary to use CUDA in the container, then navigate to the ```docker``` folder, and execute the command
```
docker build -t baxter-moveit-docker .
```
Once the container has been built, navigate the root directory, and run the ```run image.bash``` executable. This will run the docker image and open a new terminal inside the container, and this repository and all its source code for running the experiments will be mounted inside the image. All the scripts and environments can then be run inside the container (see below for details).

In the terminal opened after launching the image, follow the below steps to set up the MoveIt! planning environment.
```
source devel/setup.bash
catkin build
roslaunch baxter_mpnet_experiments baxter_mpnet.launch
```

Then in a new terminal, enter the container with the command (replace $CONTAINER_NAME with whatever name was assigned to the container that was just started)
```
docker exec -it $CONTAINER_NAME bash
```
and once inside the container, 
```
source devel/setup.bash
roscd baxter_mpnet_experiments
```
And all the MPNet scripts can be run as described in the section above. 

<!-- 
# Setting Up Experiments
The main script, ```motion_planning_data_gen.py``` uses the MoveIt Python API for setting up the environment and creating motion plan requests. The program can be used with the default MoveIt OMPL motion planners as is. To use non-default OMPL planners with the Baxter MoveIt interface, this can be done by modifying the ```planning_context_manager.cpp``` file in the ```moveit_planners_ompl``` package to include the necessary OMPL headers and register the planner in the ```registerDefaultPlanners()``` function. Then in the ```baxter_moveit_config``` package, the file ```config/ompl_planning.yaml``` file can be modified to configure the planner and apply it as the default planner (using BIT* as an example):

```
planner_configs:
  BITStarkConfigDefault:
    type: geometric::BITstar
...
right_arm:
  default_planner_config: BITStarkConfigDefault
```

after making any of these changes rebuild your ROS workspace with ```catkin build```.

 The filename to save path data to should be configured in the ```main()``` loop of the Python program, 

```python
pathsFile = "data/path_data_example"
```

along with other experiment configuration such as MoveGroup planning timeout

```python
max_time = 300
group.set_planning_time(max_time)
```

or the condition for ending data collection (such as number of total planning attempts)
```python
while (total_paths < 30): #run until either desired number of total or feasible paths has been found
    ...
```

# Environments
The environment meta-data is saved in the pickled file ```env/trainEnvironments.pkl``` and the .STL files for the obstacles (book, soda can, mug, and bottle) are save in the ```meshes/``` directory. The environment data includes the dimensions, z-offset, workspace locations, and default mesh file path for loading the scene. A table planning scene interface is included in the script which loads this environment meta data and applies the different environments to the MoveIt scene such that the MoveIt collision checker and planner can be used with these obstacles in their respective locations. For each environment, there is also a set of collision-free configurations which resemble a grasp near the table surface saved in the pickle file ```env/trainEnvironments_testGoals.pkl``` which are similarly loaded in the main script to sample from when creating planning requests. 

# Running Experiments and Analyzing Data
The simulated robot and general MoveIt environment can be set up by launching
```
roslaunch baxter_moveit_experiments baxter_moveit.launch
```
and then the Python script ```motion_planning_data_gen.py``` can be run with a ROS node name as a single command line argument to set up the motion planning experiment with the various environments,
```
python motion_planning_data_gen.py test
```

The path planning data for each environment, including the paths, planning time, path cost (C-space euclidean length), and number of successful/total planning requests are recorded in a dictionary and periodically saved in the ```data/``` folder to be analyzed or played back on the robot. ```comparison.ipynb```  in ```analysis/``` and the ```playback_path.ipynb``` notebooks are simplified examples of using the saved planning data for data analysis or visualizing the paths on the robot using the Baxter interface (ensure the robot is enabled before playing back paths, with ```rosrun baxter_tools enable_robot.py -e``` in the terminal). -->
