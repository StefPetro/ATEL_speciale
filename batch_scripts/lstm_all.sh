#!/bin/sh 
bsub < ./batch_scripts/LSTM_scripts/lstm_Genre.sh
bsub < ./batch_scripts/LSTM_scripts/lstm_Tekstbaand.sh
bsub < ./batch_scripts/LSTM_scripts/lstm_Fremstillingsform.sh
bsub < ./batch_scripts/LSTM_scripts/lstm_Semantisk_univers.sh
bsub < ./batch_scripts/LSTM_scripts/lstm_Stemmer.sh
bsub < ./batch_scripts/LSTM_scripts/lstm_Perspektiv.sh
bsub < ./batch_scripts/LSTM_scripts/lstm_Holistisk_vurdering.sh
