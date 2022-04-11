# SemEval2022-TeamLRL_NC
Repository for submission of Team LRL_NC at SemEval22-Task4:Patronizing and Condescending Language Detection

This file contains the code for the paper presented by Team LRL_NC at SemEval Task 4: Patronizing and Condescending Language Detection.

The dataset is publicly available at https://github.com/Perez-AlmendrosC/dontpatronizeme

Some other utility functions can be installed from the above repository as well.

For using the given codes, Python 3.7.12 is used and the packages used are given in `requirements.txt`.

Using these functions, and dataset, the data for generating predictions for these tasks can be prepared using the file `dataprep.py`.

The MyRoberta and MyXLNet models which represent the further pre-trained Roberta and XLNet models, can be prepared using the files `further_pretraining_roberta.py` and `further_pretraining_xlnet.py` respectively.

The systems A and B of binary classification subtask can be run using `binary_classification_systemA.py`  and `binary_classification_systemB.py`, respectively.

The systems A and B of multi-label classification subtask can be run using `multi_classification_systemA.py`  and `multi_classification_systemB.py`, respectively.
    

Official Task Website: https://sites.google.com/view/pcl-detection-semeval2022/
