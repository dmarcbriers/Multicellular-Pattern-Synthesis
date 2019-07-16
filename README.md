# Multicellular Pattern Synthesis

## Container Contributors (alphabetical order)
- [Demarcus Briers](https://github.com/dmarcbriers)
- [Iman Haghighi](https://github.com/imanh10)
- David Joy
- [Ashley Libby](https://github.com/arglibby1650G)

## Quicky Start Guide
This repository contains the code to replicate the computational environement for the publication by Demarcus Briers and Ashley Libby, Iman Haghighi, David Joy, Bruce Conklin, Calin Belta, and Todd McDevitt.

We provide a docker container to replicate our computaitonal environment on Windows, Mac, or Linux. Please see the `docker` folder for instructions on how to download or build the docker image. This is the quickest and most portable method of installing all software dependencies.

Once you have a working copy of the docker image there are three types of analyis you can perform:
* **Simulations:** For running the simulations with the computational model see the `model` folder.
* **Automated Pattern Design:** For automating the deisgn of multicellular patterns with machine learning and Particle Swarm Optimization (PSO) see the `synthesis` folder
* **Quantitative Pattern Verification:** For quantifying patterns produced by simulations or experiements see the `synthesis` folder for a Machine Learning approach and the `image_segmentation_clustering` folder for a Computer Vision approach.
