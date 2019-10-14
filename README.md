# Multicellular Pattern Synthesis

## Container Contributors (alphabetical order)
- [Demarcus Briers](https://github.com/dmarcbriers)
- [Iman Haghighi](https://github.com/imanh10)
- [David Joy](https://github.com/david-a-joy)
- [Ashley Libby](https://github.com/arglibby1650G)

## Quicky Start Guide
This repository contains the code to replicate the computational environement for the publication by Demarcus Briers and Ashley Libby, Iman Haghighi, David Joy, Bruce Conklin, Calin Belta, and Todd McDevitt.

We provide a docker container to replicate our computaitonal environment on Windows, Mac, or Linux. Please see the `docker` folder for instructions on how install the required dependencies. Installing the dependencies using Docker is the quickest and most reproducible method of installing all software dependencies.

Once you have a working copy of the docker image (or have manually installed the dependencies), there are three types of analyis you can perform:
* **Simulations:** For running the simulations with the computational model see the `model` folder.
* **Automated Pattern Design:** For automating the deisgn of multicellular patterns with supervised learnings(TSSL) and mathematical optimization (Particle Swarm Optimization), see the `synthesis` folder. 
* **Quantitative Pattern Verification:** For quantifying patterns produced by simulations or experiements see the `synthesis` folder for a Machine Learning approach and the `image_segmentation_clustering` folder for a Computer Vision approach.
