# Author: Eric Reyes SJSU

## Installation Instructions
There are a lot of dependencies for this project and the fastest way to get setup with GPU support is through docker. 

> Installation instructions for docker can be found [here](https://docs.docker.com/desktop/windows/install/)

Once installed, download my repository from https://github.com/Juxsta/diy-alexa. 
Read the original read me and download the speech_commands file from google and extract the files to the model folder under the name "speech_data". 

Now open the terminal and change directories into the project
```
cd /{Download Directory}/diy-alexa/
```
Use this command to mount the project into the docker container.
```
docker run --gpus all -d -it -p 8848:8888 --name gpu-jupyter -v ${PWD}:/home/jovyan/work -e GRANT_SUDO=yes -e JUPYTER_ENABLE_LAB=yes --user root juxsta/gpu-jupyter:latest
```
This will pull my docker image with the precompiled dependencies for this project and also setup a jupyter-notebook server.
The download and extraction will take a while as the image is about 15GB.
By default the jupyter notebook server can be accessed from port 8888. I.e. open the browser and navigate to `http://localhost:8888`. 

From there you can access the scripts in the project and all the necessary dependencies should already be downloaded. 
If using a gpu this block will tell you if the gpu can be accessed properly
```
tf.test.is_built_with_cuda()
tf.config.list_physical_devices('GPU')
``` 
Proper output 
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```
No GPU detected output
```
[]
```