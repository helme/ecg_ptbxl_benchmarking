#!/bin/bash

####################################
#   GET PTBXL DATABASE
####################################
mkdir -p data
cd data
wget https://storage.googleapis.com/ptb-xl-1.0.1.physionet.org/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip
unzip ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip
mv ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1 ptbxl
cd ..

####################################
#   GET ICBEB CHALLGENGE DATABASE
####################################
mkdir -p tmp_data
cd tmp_data
wget http://2018.icbeb.org/file/REFERENCE.csv
wget http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet1.zip
wget http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet2.zip
wget http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet3.zip
unzip TrainingSet1.zip
unzip TrainingSet2.zip
unzip TrainingSet3.zip
cd ..
python code/utils/convert_ICBEB.py
cp data/ptbxl/scp_statements.csv data/ICBEB/
