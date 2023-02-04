#!/bin/sh 

list="1 2 3 4 5 6 7 8 9 10"
target="Genre Tekstbaand Fremstillingsform Semantisk_univers Stemmer Perspektiv Holistisk_vurdering"
acq_function="entropy random"

for a in $acq_function
do
    for l in $list
    do
        for t in $target
        do
            bsub < ./batch_scripts/AL_scripts/$a/$t/AL_${t}_cv$l.sh
        done
    done
done

for a in $acq_function
do
    for l in $list
    do
        for t in $target
        do
            bsub < ./batch_scripts/AL_scripts/$a/$t/AL_${t}_cv$l.sh
        done
    done
done