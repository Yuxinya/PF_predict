## Power Factor Prediction of Thermoelectric Materials Using Machine Learning

## Premise

This is a random forest machine learning model for power pactor prediction of thermoelectric materials. 

Developed in 2022.4-2 at <br />
School of Mechanical Engineering<br />
Guizhou University, Guiyang, China <br />

## Environment Setup

To use this machine learning model, you need to create an environment with the correct dependencies. Using `Anaconda` this can be accomplished with the following commands:

```bash
conda create --name PF_predict python=3.6
conda activate PF_predict
conda install --channel conda-forge pymatgen
pip install matminer
pip install scikit-learn==0.24.1
```

## Setup

Once you have setup an environment with the correct dependencies you can install by the following commands:

```bash
conda activate PF_predict
git clone https://github.com/Yuxinya/PF_predict
cd PF_predict
pip install -e .
```
