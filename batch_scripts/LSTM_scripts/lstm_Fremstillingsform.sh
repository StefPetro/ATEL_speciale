#!/bin/sh 

list="1"

for l in $list
do
    bsub < ./batch_scripts/LSTM_scripts/Fremstillingsform/lstm_Fremstillingsform_cv$l.sh
done
