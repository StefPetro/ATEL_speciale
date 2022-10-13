#!/bin/sh 
bsub < ./BERT_scripts/BERT_finetune_Genre.sh
bsub < ./BERT_scripts/BERT_finetune_Tekstbaand.sh
bsub < ./BERT_scripts/BERT_finetune_Fremstillingsform.sh
bsub < ./BERT_scripts/BERT_finetune_Semantisk_univers.sh
bsub < ./BERT_scripts/BERT_finetune_Stemmer.sh
bsub < ./BERT_scripts/BERT_finetune_Perspektiv.sh
bsub < ./BERT_scripts/BERT_finetune_Holistisk_vurdering.sh
