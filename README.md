# 104182850-BigDataProject
 PMJayasinghe Big Data Project - Detecting Financial Fraud with Machine Learning and Deep Learning Algorithms


## Step 1

Download the dataset from the following link:
https://www.kaggle.com/datasets/ealaxi/paysim1/data

unzip the file and move the file to the data folder

## Step 2

setup the environment using venv

```bash
python -m venv venv
```
Please note that this project was developed using python 3.10 on an Apple M3 Max chip.
If you are using different hardware, you may need to install the dependencies manually.

## Step 3

activate the venv

```bash
source venv/bin/activate
```

install the dependencies

```bash
pip install -r requirements.txt
```

## Step 4

run the first script to preprocess the data

```bash
python scripts/preprocessing.py
```

## Step 5

now run the classification models 

```bash
python scripts/ml_classification.py
python scripts/dl_classification.py
```

## Step 6

finally, run the prediction scripts

```bash
python scripts/ml_prediction.py 
python scripts/dl_prediction.py
```
