#!/bin/sh 
bsub < ./batch_scripts/RoBERTa_scripts/roberta_Fremstillingsform.sh
bsub < ./batch_scripts/RoBERTa_scripts/roberta_Genre.sh
bsub < ./batch_scripts/RoBERTa_scripts/roberta_Holistisk_vurdering.sh
bsub < ./batch_scripts/RoBERTa_scripts/roberta_Perspektiv.sh
bsub < ./batch_scripts/RoBERTa_scripts/roberta_Semantisk_univers.sh
bsub < ./batch_scripts/RoBERTa_scripts/roberta_Stemmer.sh
bsub < ./batch_scripts/RoBERTa_scripts/roberta_Tekstbaand.sh