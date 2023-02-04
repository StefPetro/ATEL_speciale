#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set the job Name -- 
#BSUB -J LSTM_Stemmer
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=4GB]"
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start -- 
##BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -oo ./out_files/LSTM/LSTM_Stemmer.out
#BSUB -eo ./out_files/LSTM/LSTM_Stemmer.err


# here follow the commands you want to execute 

# Load dependencies
module load python3/3.9.14
#module load gcc/10.3.0
#module load cuda/11.5
module load cudnn/v8.3.0.98-prod-cuda-11.5

source ./venv/bin/activate
# echo $VIRTUAL_ENV

python3 run_lstm.py --target_col "Stemmer" --cv 1
