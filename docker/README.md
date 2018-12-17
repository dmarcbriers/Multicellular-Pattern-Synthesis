# Multicellular Pattern Synthesis Docker

## Short Description

This container replicates the computational environemet for the publication Briers & Libby et al.

## Container Contributors
- [Demarcus Briers](https://github.com/dmarcbriers)
- Iman Haghighi
- David Joy
- Ashley Libby


## Quickstart
If you already have docker installed on your computer you can run the following commands to pull a docker image from dockerhub. Make sure are in the root folder of the repository to have access to all files from inside the docker container.
```docker pull dmarcbriers/mc-pattern-synthesis:latest ```
```docker run --rm -v ./:/home/ubuntu/ dmarcbriers/mc-pattern-synthesis:latest -it /bin/bash```

This will launch an interactive terminal with all necessary software. See the README in the respective directiory for further instructions.

## Build Instructions
If a ready made image was not avaible from dockerhub, you can quickly rebuild the image using the command (make sure you are inside of the docker directory when running the build command):
``` docker build --tag="dmarcbriers/mc-pattern-synthesis:latest" .```
```cd ../ ```
```docker run --rm -v ./:/home/ubuntu/ dmarcbriers/mc-pattern-synthesis:latest -it /bin/bash```