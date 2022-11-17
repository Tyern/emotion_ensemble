Folder structure:
- Emotion_Ensemble
|
|- Images
||- anger
||- disgust
||- ...
|
|- Labels
|| ground_truth_test.csv
|| ground_truth_train.csv
|
|- src
|-- ...


## train:
python train_test_split.py
python train.py

## predict and create csubmission_result.csv file:
python predict.py <weight>.pth