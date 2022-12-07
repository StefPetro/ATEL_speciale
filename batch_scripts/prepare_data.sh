#!/bin/sh 


# Load dependencies
module load python3/3.9.14

python3 -m pip install --user -r requirements.txt

# module load matplotlib/3.4.2-numpy-1.21.1-python-3.9.6
module load pandas/1.3.1-python-3.9.14
### General options 
### -- specify queue -- 
#BSUB -q gpua40
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set the job Name -- 
#BSUB -J prepare_data
### -- ask for number of cores (default: 1) -- 
#BSUB -n 1
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=100GB]"
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o batch_out/prepare_data.out
#BSUB -e batch_out/prepare_data.err

# here follow the commands you want to execute 
python3 prepare_dataset.py
