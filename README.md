# CCS Model Training and Prediction

Publication:
- doi: https://doi.org/10.1101/2020.05.19.102285
- biorxiv: https://www.biorxiv.org/content/10.1101/2020.05.19.102285v1

## Library Setup

Setup CUDA 10.0 with cudnn and install the required python libraries with pip:

```
pip install -r requirements.txt
```

## Prediction with Pre-Trained Model

Unzip the checkpoint found in out.

Prepare a csv file that contains Sequence and Charge Information and use the provided `predict.py` script:
```
python predict.py <filename.csv> 
```
For the format see the provided example file in `./data/combined_reduced.csv`

## Process data
Use the provided notebook: `process_data_final.ipynb`

It uses the raw data files and saves train and test files in pkl format to disc in `./data_final`

## Training

The `bidirectional_lstm.py` file contains training and prediction routines.

Training is done by setting the paths in `run_training.py` and executing it.
The complete dataset will be uploaded at a later stage of publication.

## Evaluation

Use the provided `evaluate.ipynb` Jupyter Notebook.

