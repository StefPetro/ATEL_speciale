#!/bin/sh 
bsub < ./ATEL_speciale/BERT_scripts/BERT_finetune_Genre.sh
bsub < ./ATEL_speciale/BERT_scripts/BERT_finetune_Tekstbaand.sh
bsub < ./ATEL_speciale/BERT_scripts/BERT_finetune_Fremstillingsform.sh
bsub < ./ATEL_speciale/BERT_scripts/BERT_finetune_Semantisk_univers.sh
bsub < ./ATEL_speciale/BERT_scripts/BERT_finetune_Stemmer.sh
bsub < ./ATEL_speciale/BERT_scripts/BERT_finetune_Perspektiv.sh
bsub < ./ATEL_speciale/BERT_scripts/BERT_finetune_Holistisk_vurdering.sh
