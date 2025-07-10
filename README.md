# Brian_3D_simulation

run_forward_3dnodes.py is the main program script for running 3D brain ultrasound simulations. In this script, we can specify the positions of the excitation source and the receiver, set the corresponding material parameters, and define the running mesh file. 

The "myjob.pbs" file is a script used for parallel computing on a computer cluster. You can submit computing tasks by using the command "qsub myjob.pbs". 

The core code for calculation and the related environment configuration have been packaged into a Singularity image and uploaded. In the paper, the grid files and speed files used for calculation have also been uploaded. The corresponding files can be found by readers on the website https://doi.org/10.5281/zenodo.15836423.
