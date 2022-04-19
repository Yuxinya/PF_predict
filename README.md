## Power Factor Prediction of Thermoelectric Materials Using Machine Learning


## Premise

This is a random forest machine learning model with a new feature set combined with the standard composition features such as Magpie descriptors for effective space group prediction for inorganic materials. 

Developed in 2021.4-2 at <br />
School of Mechanical Engineering<br />
Guizhou University, Guiyang, China <br />

Machine Learning and Evolution Laboratory<br />
Department of Computer Science and Engineering<br />
University of South Carolina, Columbia, USA<br />


## Performance on Materials Project dataset

Our model of space group prediction in cubic material is trained with the dataset of 'ML/cubic.csv' by useing the 'ML/RF_of_us.py'
, and the dataset used for other crystal system training can be downloaded from here [data.csv](https://figshare.com/s/9cfe81a3b087618353c8).
Moreover, the two previous work frameworks for space group classification are also put in the ML folder.

Prediction performance for space groups over different crystal systems （10 fold cross validation)
|Crystal system|data set size |   Accuracy  |     MCC     |   Precision |   Recall  |   F1 score  |
|-------------|---------------|-------------|-------------|-------------|-----------|-------------|
Cubic         |     17367     | 0.961±0.006 | 0.945±0.008 | 0.960±0.005 |0.961±0.006| 0.959±0.006 |
Hexagonal     |      8201     | 0.909±0.008 | 0.888±0.010 | 0.908±0.008 |0.909±0.008| 0.906±0.008 |
Trigonal      |      9429     | 0.824±0.012 | 0.797±0.014 | 0.823±0.013 |0.824±0.012| 0.818±0.012 |
Tetragonal    |     12675     | 0.849±0.013 | 0.832±0.015 | 0.846±0.013 |0.849±0.013| 0.840±0.014 |
Orthorhombic  |     22392     | 0.755±0.005 | 0.729±0.006 | 0.759±0.005 |0.755±0.005| 0.746±0.006 |
Monoclinic    |     23024     | 0.712±0.009 | 0.647±0.011 | 0.715±0.010 |0.712±0.009| 0.703±0.010 |
Triclinic     |      9440     | 0.835±0.013 | 0.665±0.026 | 0.835±0.013 |0.835±0.013| 0.834±0.013 |
<!--- img src="performance1.png" width="800"--->

## Environment Setup

To use this machine learning model, you need to create an environment with the correct dependencies. Using `Anaconda` this can be accomplished with the following commands:

```bash
conda create --name SG_predict python=3.6
conda activate SG_predict
conda install --channel conda-forge pymatgen
pip install matminer
pip install scikit-learn==0.24.1
```

## Setup

Once you have setup an environment with the correct dependencies you can install by the following commands:

```bash
conda activate SG_predict
git clone https://github.com/Yuxinya/SG_predict
cd SG_predict
pip install -e .
```
