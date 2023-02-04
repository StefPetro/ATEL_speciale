#!/bin/sh 
bsub < ./batch_scripts/BERT_scripts/BERT_finetune_Genre.sh
bsub < ./batch_scripts/BERT_scripts/BERT_finetune_Tekstbaand.sh
bsub < ./batch_scripts/BERT_scripts/BERT_finetune_Fremstillingsform.sh
bsub < ./batch_scripts/BERT_scripts/BERT_finetune_Semantisk_univers.sh
bsub < ./batch_scripts/BERT_scripts/BERT_finetune_Stemmer.sh
bsub < ./batch_scripts/BERT_scripts/BERT_finetune_Perspektiv.sh
bsub < ./batch_scripts/BERT_scripts/BERT_finetune_Holistisk_vurdering.sh
