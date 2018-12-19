# Multicellular Pattern Synthesis Docker

## Short Description

This container replicates the computational environemet for the publication: Briers, Libby,Haghighi, Joy, Conklin, Belta, and McDevitt

## Container Contributors (alphabetical order)
- [Demarcus Briers](https://github.com/dmarcbriers)
- Iman Haghighi
- David Joy
- Ashley Libby


## Quickstart
If you already have docker installed on your computer you can run the following commands to pull a docker image from dockerhub. If you are not familiar with docker there is a cheetsheet [here](https://github.com/wsargent/docker-cheat-sheet): To pull the image from dockerhub make sure your create a free account, then enter you login credential from the command line using:

```docker login -u <username> -p```

Once you have entered you login credential on the docker command line, make sure you are in the root folder of the repository to have access to all files from inside the docker container.

```docker pull dmarcbriers/mc-pattern-synthesis:latest ```

If that command was sucessful you can launcha docker container using the command below. Note, sometimes being on a VPN prevents docker from working so try disconnecting from your VPN client if docker is not working.

```docker run --rm -v ${PWD}:/home/ubuntu/:rw dmarcbriers/mc-pattern-synthesis:latest -it /bin/bash```

This will launch an interactive terminal with all necessary software. See the README in the model directiory for further instructions on running simulations.

## Build Instructions
If a ready made image was not avaible from dockerhub, you can quickly rebuild the image using the command (make sure you are inside of the docker directory when running the build command, and dont forget the period at the end):

```docker build --tag="dmarcbriers/mc-pattern-synthesis:latest" .```

```cd ../model/ ```

```docker run --rm -v ${PWD}:/home/ubuntu/:rw dmarcbriers/mc-pattern-synthesis:latest -it /bin/bash```

Now you can pull up the help documentation to run a simulation with:
```python3 morpheusSetup.py -h ```
