
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import bz2 
import pickle 
import _pickle as cPickle
import os
import joblib

# x = pd.read_csv('Data/data.csv').iloc[106953:,11:].fillna(0).values #Cubic
# y = pd.read_csv('Data/data.csv').iloc[106953:,4].fillna(0).values #137 255
x = pd.read_csv('Data/data.csv').iloc[:,14:].fillna(0).values #1-2 Triclinic
y = pd.read_csv('Data/data.csv').iloc[:,3].fillna(0).values
# x = pd.read_csv('Data/data.csv').iloc[15297:45169,11:].fillna(0).values #3-15 Monoclinic
# y = pd.read_csv('Data/data.csv').iloc[15297:45169,6].fillna(0).values
# x = pd.read_csv('Data/data.csv').iloc[45169:71970,11:].fillna(0).values #16-74 Orthorhombic
# y = pd.read_csv('Data/data.csv').iloc[45169:71970,6].fillna(0).values 
# x = pd.read_csv('Data/data.csv').iloc[71970:86624,11:].fillna(0).values #75-142 Tetragonal
# y = pd.read_csv('Data/data.csv').iloc[71970:86624,6].fillna(0).values 
# x = pd.read_csv('Data/data.csv').iloc[86624:97710,11:].fillna(0).values #143-167 Trigonal
# y = pd.read_csv('Data/data.csv').iloc[86624:97710,6].fillna(0).values
# x = pd.read_csv('Data/data.csv').iloc[97710:106953,11:].fillna(0).values #169-194 Hexagonal
# y = pd.read_csv('Data/data.csv').iloc[97710:106953,6].fillna(0).values
# x = pd.read_csv('Data/data.csv').iloc[:,11:].fillna(0).values #Crystal
# y = pd.read_csv('Data/data.csv').iloc[:,6].fillna(0).values
# print(x)
y = y.reshape(-1,1)
x_y = np.hstack((x,y))
np.random.shuffle(x_y)
x_train = x_y[:,:-1]
y_train = x_y[:,-1]
# print(x_train)
# print(x_train.shape)

forest=RandomForestRegressor(#n_estimators=50,
							 # min_samples_split=10,
                             #criterion='mse',
                             # min_samples_leaf=2,
                             # random_state=i,
                             n_jobs=-1)

forest.fit(x_train,y_train)


y_pred=forest.predict(x_train)
r2 = r2_score(y_train , y_pred)
print(r2)
print(y_pred)

dirs = 'Model'
if not os.path.exists(dirs):
    os.makedirs(dirs)
def compressed_pickle(title, data):
	with bz2.BZ2File(title + '.pbz2', 'w') as f: 
		cPickle.dump(data, f)
compressed_pickle(dirs+'/PFpredict', forest)
def decompress_pickle(file): 
	data = bz2.BZ2File(file, 'rb') 
	data = cPickle.load(data) 
	return data
forest = decompress_pickle(dirs+'/PFpredict.pbz2')
y_pred = forest.predict(x_train)
r2 = r2_score(y_train , y_pred)
print(r2)
print(y_pred)
