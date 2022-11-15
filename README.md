# SER
An hierarchical approach for multilingual speech emotion recognition (SER)

Regarding the YAMNet files they can be found here: https://github.com/LIMUNIMI/HAR-YAMNet but two files from this repository mus be changed with the ones here: https://github.com/NicoRota-0/SER/tree/main/YAMNET_FILES

Regarding the dataset, it is possible to download it from here: https://drive.google.com/file/d/1Hj3Xz-fS2uY09aAc5U9iWiFfewTKLQ_Z/view?usp=share_link
(and place it in the same working directory of the project)

First unzip the compressed files, then run the python notebook in the dataset folder, then it is possible to train and test the three models:

- k-NN (first load the data with knn_load_data.m file, then evaluate the model with eval_knn.m file)
- YAMNet (from https://github.com/LIMUNIMI/HAR-YAMNet)
- BiLSTM (BiLSTM.m file)
