# Trajectory Planning
The goal of this project is to train a neural network for driving a car in the [Udacity self-driving simulator](https://github.com/udacity/self-driving-car-sim) and help it keep on the road.

## Prerequisites
- Python 3
- Tensorflow
- OpenCV

## What is in each file?
#### DataProc
- Contains the code to load and process the data from the simulator.
#### Model
- Functions to make the convolutional and fully connected layers.
- A network function which defines the neural network architecture.
#### Train
- This is the main file that is used to train the model and save it.

## Usage
The data file created by the Udacity simulator should be saved in the same directory as the python files, as in the repo. All the images should be saved in the `images` folder.
The data file should contain either the relative or absolute paths to the images.
The command to be used to train the model is as follows
`python3 Train.py --datafile <path_to_datafile> --save_dir <dir_to_save_tf_model> --summary_dir <dir_to_save_tf_summary>`
Some optional arguments with the default values in the parenthesis-
- `--num_epochs` the number of epochs (200)
- `--minibatch_size` size of the minibatch to be used (128)
- `--log_file` path to an existing or new file - a log of training is written in this file (log.txt).

## Further Work
- Switch to a better simulator, with traffic and realistic conditions.
- Use Reinforcement Learning instead of Supervised Learning.