# GP-net+: Learning to Grasp Unknown Objects in Domestic Environments

Code and data for paper "Learning to Grasp Unknown Objects in Domestic Environments", which is currently under review. 
Grasping unknown objects in domestic environments using data-driven methods requires appropriate training
data for the models. In order to realise this, this codebase can prepare furniture units,
simulate domestic scenes using those furniture units/object meshes, test grasps and store
them in a training dataset.

You can use the resulting training dataset to train grasp proposal networks, for example, our
6-DoF grasp proposal network for flexible viewpoints called GP-net+.
GP-net+ is a Fully Convolutional Neural Network (FCNN) with a ResNet-50 architecture.

## Installation

We use PyBullet for simulating the domestic scene environments and PyTorch for training 
and evaluating GP-net+.

We recommend installation using a new conda environment. You can install all the necessary
packages by running:


```conda env create -f environment.yml```


## A. Generating training data

Generating training data consists of three steps:

1. Preparing object meshes
2. Preparing furniture meshes
3. Constructing training scenes, testing grasps and saving data


You can skip the process of generating training data if you want to use the original training dataset for GP-net+,
which is available at [zenodo.org](https://zenodo.org/records/10083842).

### 1. Preparing object meshes

We used objects from the ShapeNet, YCB, BigBird and EGAD object sets in our experiments. 
The resulting object files can be found in the data zip file.
If you plan to use other object meshes for creating a simulation, you can use the following steps
to replicate our pre-processing and grasp sampling:

1. Simplify object meshes using [meshlab](https://www.meshlab.net/) (if too many vertices/edges).
Makes sure the centre of mass of the objects is sensible, since this will affect their behaviour in pybullet. 
Note that for ShapeNet objects, you need to revert the normalisation of the meshes as specified on the ShapeNet website.
2. Calculate a convex decomposition of the objects using [v-hacd](https://github.com/kmammou/v-hacd). This allows for 
a concave behaviour of objects in pybullet.
3. Sample grasps for the objects. We used the antipodal grasp sampler from Dex-Net with an adapted
sampling procedure as per our previous paper, GP-net. The code with a docker implementation of Dex-Net
can be found on [github](https://github.com/AuCoRoboticsMU/gpnet-data/blob/master/tools/sample_grasps.py)
4. Create urdf files based on the convex decomposition of the objects. You can use `src/create_urdfs.py` for this.

### 2. Preparing furniture meshes

We used 100 furniture meshes from the ShapeNet dataset in our experiments. The resulting
meshes and sampling spaces can be found in the data zip file.
If you plan to use other furniture units for creating a simulation, you can use the following steps
to replicate our pre-processing steps:

1. Revert the normalisation of the furniture meshes, simplify the mesh, set their fil their y-z axis 
and move the objects to the ground. We do this using [meshlab](https://www.meshlab.net/) and the bash script
2. Adjust the position of shelves if needed (e.g. a shelf mounted on the wall rather than standing on the ground)
3. Use [blender](https://www.blender.org/) to identify regions of workspaces using `src/preprocessing_furniture/blender_find_workspace.py`. 
We used a docker container with `blender_docker.sh` to run these scripts, specifying the appropriate paths to the .obj files
of the chosen furniture units. This creates .csv files with the coordinates of adjacent regions, they will have to be filtered
manually to exclude ridges, the bottom side of shelves and inaccessible workspaces by deleting the according rows in the .csv files.
4. Create the object workspaces files for each furniture unit which specifies where objects can be placed
during scene generation. You can use `generate_object_workspaces.py` for this, which
processes the filtered csv files with the adjacent workspace regions, creates a concave 2D region from them
and stores the results.
5. Manually add walls to furniture units as needed. We do this using [meshlab](https://www.meshlab.net/). We record the position
of walls in a single csv file like all_walls.csv in the data zip file.
6. Create the camera sampling spaces for each furniture unit. This specifies where the camera pose will be sampled during
scene generation. You can use `generate_camera_sampling_spaces.py` for generating and storing
the camera sampling spaces. Note this will use the object workspaces, object mesh and the all_walls.csv files.
7. Finally, generate the .urdf files for the furniture units using `src/create_furniture_urdfs.py`. Note that since the furniture meshes are static objects
in pybullet, they do not require a convex decomposition but can be simulated as concave objects.

### 3. Simulating training scenes


You can use `generate_training_data.py` to simulate training scenes and save the data. It expects
both the furniture meshes and the object meshes to be divided in folders according to their split (train/val/test).
In order to speed up the simulation of training scenes, you can use [multiprocessing](https://docs.python.org/3/library/multiprocessing.html).
If you need to visualise the process of generating data, you can use the `--sim-gui` option when calling the script in order
to activate pybullet's GUI.

The depth images, segmentation images, grasps and stats are saved in compressed .npz files, which can be loaded using numpy.
There is a counter in the naming convention of all of those files, e.g. `depth_image_0000123.npz`, corresponding to the 123rd
image rendered in the training data generation process.

## B. Training GP-net+

To train GP-net+, you can use `src/train_gpnetplus.py`. The progress after each epoch is stored in `data/runs`
and can be visualised using tensorboard, e.g. with `tensorboard --logdir data/runs`.

## C. Evaluating Networks

Evaluating GP-net+ and other networks on domestic scenes can be achieved in two steps: generating the 
evaluation scenes and running inference with the networks. Make sure to download `gpnetplus_simulation_data.zip`  from
[zenodo.org](https://zenodo.org/records/10083842) and unpack it in this directory to have the meshes, urdf
files and pre-trained GP-net+ model for evaluating the models available.

### Generating evaluation scenes

You can generate evaluation scenes using `generate_evaluation_scenes.py --logname $MY_LOGNAME`. By default, this will create
domestic grasping scenes (using the furniture units), but you can also choose to use a simple tabletop setup
with a planar surface and objects in a constricted workspace by setting the flag `--no-domestic-scene`.
The scenes will be stored in `data/experiments/$MY_LOGNAME`. You can specify the number of scenes to generate,
and the objects that will be used. This script also samples camera coordinates and saves them,
so that you can use them if you want to compare multiple algorithms on the same scenes.

### Running inference on evaluation scenes

Once the scenes are generated, you can run evaluation using 
`run_evaluation_scenes.py --logname $MY_LOGNAME --model $MODEL_TO_TEST`. Make sure to set the flag
`--object-set $OBJECT_SET` to the same objects used when generating the evaluation scenes. You can also vary the
detection threshold used when mapping the dense tensor output of GP-net+ to grasp proposals, e.g. using 
`--detection-threshold 0.4`.

If you want to change the fitness function, run the evaluation using a small threshold, e.g. `--detection-threshold 0.01`.
You can then use `check_fitness_function.py` to adjust the fitness function and visualise the results.

### Visualising evaluation results

You can visualise evaluation results using `visualise_evaluation_results.py` by changing the variable `experiment_name`
to your `$MY_LOGNAME` variable. You can modify the script for custom evaluations if required.


-----------------------

If you use this code for your research, please consider citing:

A. Konrad, J. McDonald and R. Villing, "Learning to Grasp Unknown Objects in Domestic Environments," in review.


This publication has emanated from research supported in part by Grants from Science Foundation Ireland under 
Grant numbers 18/CRT/6049 and 16/RI/3399.
The opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do 
not necessarily reflect the views of the Science Foundation Ireland.
