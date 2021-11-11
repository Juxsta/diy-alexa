# Author: Eric Reyes SJSU

## Installation Instructions
There are a lot of dependencies for this project and the fastest way to get setup with GPU support is through docker. 

> Installation instructions for docker can be found [here](https://docs.docker.com/desktop/windows/install/)

Once installed, download my repository from https://github.com/Juxsta/diy-alexa. 
Read the original read me and download the speech_commands file from google and extract the files to the model folder under the name "speech_data". 

## Using Docker 

I simplified the usage steps for docker and created a custom dockerfile included here. As long as docker is working properly all that needs to be done is to cd into this directory and run docker compose up. 

```
cd {Download Directory}/diy-alexa/model
docker compose up
```
This will setup the environment and get a jupyter notebook running that can be immediately used to run the project. As a bonus, as long as the latest drivers for your gpu are installed (they probably are "geforce game ready drivers are enough") then passthrough should be flawless.