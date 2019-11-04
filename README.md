# HMM-POS-Tagger
Here you can find two different POS tagging models: 1. Baseline 2: HMM Viterbi.
The baseline models basically counts the tags assigned to each and considers the
 most frequent one as the correct tag for that word.
 # 
 To run the baseline model use the command below:

 ```bash
python3 baseline.py path_to_train_data path_to_test_data
```

In order to run Viterbi use the following command:

 ```bash
python3 postagger.py path_to_train_data path_to_test_data
```