#!/bin/sh 

list="1 2 3 4 5 6 7 8 9 10"
target="Genre Tekstbaand Fremstillingsform Semantisk_univers Stemmer Perspektiv Holistisk_vurdering"

for l in $list
do
    for t in $target
    do
        bsub -n ./batch_scripts/LSTM_scripts/$t/lstm_${t}_cv$l.sh
    done
done
