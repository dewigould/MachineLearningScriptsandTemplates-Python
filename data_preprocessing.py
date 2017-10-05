# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values    #matrix of indepedent variables
y = dataset.iloc[:, 3].values        #vector of dependent variables

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[:,1:3])   #only fit this to the columns with missing data (1,3 because last column isn't included)

X[:,1:3] = imputer.transform(X[:,1:3])

#Encoding categorical data (give data like country name, or person name a NUMBER)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()      #dealing with country column first
X[:,0] = labelencoder_X.fit_transform(X[:,0])    #encode first column to numbers instead of names
onehotencoder = OneHotEncoder(categorical_features = [0])    #categorical features tells it which column to deal with
X = onehotencoder.fit_transform(X).toarray()


#program knows that this column is already categorical so don't need to use onehotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#splitting dataset into Training Set and Test Set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,random_state = 0)   #random_state just so that I get same results as teacher, not important

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)    #need toFIT to training set, and then TRANSFORM
X_test = sc_X.transform(X_test)      #don't need to FIT as it's already FIT to TRAINING SET

