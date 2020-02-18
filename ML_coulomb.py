#don't forget scaling and k-fold cross validation
# suggestion to add keras deep learning
#IMPORTS
import numpy as np
import pandas as pd
import pickle
from pprint import pprint as pp
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
torch.__version__
torch.cuda.is_available()
import xgboost as xgb
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# machine learning
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import OneHotEncoder
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron


from tensorflow import keras
from keras import Sequential
from tensorflow.keras import layers
from keras.layers import Dense  #Dense means fully connected layers
from keras import metrics


sns.set()
%matplotlib inline

coulomb_df.shape

atom_types_df = pd.read_pickle("atom_types.pic")
# print(atom_types)
coulomb_df = pd.read_pickle("coulomb_interactions.pic")
# print(coulomb_df)
TI_df = pd.read_pickle("TI.pic")
print(TI_df)
TI_df.max()
TI_df.min()

preds = pd.read_pickle("TI_test_preds.pickle")
preds.max()
preds.min()
coulomb_df = StandardScaler().fit_transform(coulomb_df)

x = coulomb_df
# x_enc = OneHotEncoder(categories='auto').fit_transform(x)
y = np.log10(TI_df)
x.shape
y.shape
# label_enc = LabelEncoder()
# x_enc = label_enc.fit_transform(x)
# y_enc = label_enc.fit_transform(y)
# x_enc = label_enc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


#principal component analysis
# Plot data to explore different properties

#one can immediately obtain that there is a clear quadratic path and the values are constrainted.
#there is a lot of great potential to minimize the amount of features
#maybe due to normalization

# Principal Component Analysis

pca = PCA(n_components=60)
pca.fit(x, y)
print(pca.explained_variance_ratio_)

x_pca = pca.transform(x)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

print(x_train_pca.shape, x_test_pca.shape)
plt.plot(x_pca, y)


#already two components give insight about how the data is structured, increasing
#the number of components would not give a lot more further insight
#to find the best values we plot the variance against components

# how many PCAs are enough

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')
plt.title('Explained Variance')
plt.show()

# playing with the number of components shows that 60 components gives you over
#0.8% variance, which is enough

# Linear Regression --------------------
# Linear Regression --------------------
linreg = LinearRegression()
linreg.fit(x_train, y_train)
Y_pred = linreg.predict(x_test)
# Compute the rmse from sklearns metrics module imported earlier
rmse = np.sqrt(mean_squared_error(y_test, Y_pred))
print("RMSE: %f" % (rmse))

linreg = LinearRegression()
linreg.fit(x_train_pca, y_train_pca)
Y_pred = linreg.predict(x_test_pca)
# Compute the rmse from sklearns metrics module imported earlier
rmse = np.sqrt(mean_squared_error(y_test_pca, Y_pred_pca))
print("RMSE: %f" % (rmse))


kfold = KFold(n_splits=10, random_state=7)
results = -1*cross_val_score(linreg, x, y, cv=kfold, scoring='neg_mean_squared_error')
# Note to self:
# The unified scoring API always maximizes the score, so scores which need to be minimized are negated in order for the unified scoring API to work correctly.
# The score that is returned is therefore negated when it is a score that should be minimized and left positive if it is a score that should be maximized.
results
results.mean()
print(f"mean_squard_error: {results.mean()}\nstandard_deviation: {results.std()}")
plt.rcParams["figure.figsize"] = (10,10)
plt.scatter(y_test,Y_pred)

## XGBoost ----------------------
#booster [default=gbtree] change to gblinear to see. gbtree almost always outperforms though
# xgboost = xgb.XGBClassifier(max_depth=14, n_estimators=1000, learning_rate=0.05,colsample_bytree=1)  #hyperparams
xgboost = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators =10)
xgboost.fit(x_train, y_train)
Y_pred= xgboost.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, Y_pred))
print("RMSE: %f" % (rmse))
plt.rcParams["figure.figsize"] = (10,10)
plt.scatter(y_train,xgboost.predict(x_train))

# Below the crossvalidation was performed on entire dataset
# However better is first perform k-fold cross validation on training data
# After review its performance  on test dataset

#-------k-fold Cross Validation using XGBoost-------
# XGBoost supports the k-fold cross validation with the cv() method
# nfolds is number of cross-validation sets to be build
# More parameters in XGBoost API reference: https://xgboost.readthedocs.io/en/latest/python/python_api.html

#Create Hyper Parameter dictionary params and exclude n_estimators and include num_boost_rounds
params = {"objective":"reg:squarederror",'colsample_bytree': 0.3, 'learning_rate': 0.1,'max_depth': 5, 'alpha': 10}

# nfold is 3, so three round cross validation set using XGBoost cv()
# cv_results include the train and test RMSE metrics for each round
# Separate targetvariable and the rest of the variables

# Convert to optimized data-structure, which XGBoost supports
data_dmatrix = xgb.DMatrix(data=x_train,label=y_train)
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3, num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)
cv_results.head()
cv_results.tail()
# Print Final boosting round metrics
# The final result may depend upon the technique used, so you may want to try different
# e.g. grid search, random search Bayesian optimization
rmse = np.sqrt(mean_squared_error(y_test, Y_pred))
print("RMSE: %f" % (rmse))
print((cv_results["test-rmse-mean"]).tail(1))


## Neural Network with Keras
#based off https://www.freecodecamp.org/news/how-to-build-your-first-neural-network-to-predict-house-prices-with-keras-f8db83049159/


#I'm also messing with the hyperparms as I go. Here's some notes so not to try things many times
#sequential layer, relu, relu, sigmoid works best
#softmax->relu

# tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.

# define the keras model
from tensorflow.keras import layers
model = Sequential()
model.add(Dense(12, input_shape=((900,)), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# model = Sequential([ Dense(40, activation='relu', input_shape=(900,)),  Dense(32, activation='relu'), Dense(1, activation='sigmoid')])
# model.compile(optimizer='sgd',   loss='binary_crossentropy', metrics=['accuracy'])
#adam is stochastic gradient descent algorithm
# model.compile(optimizer='adam',   loss='binary_crossentropy', metrics=['accuracy'])
model.compile(loss='mean_squared_error',optimizer='sgd',metrics=[metrics.mae])
)
hist = model.fit(x_train, y_train, batch_size=64, epochs=100,validation_data=(x_train,y_train),verbose=0)
# model.evaluate(X_test, Y_test)[1]
Y_pred = model.predict(x_test)
loss_NN = model.evaluate(x_test, y_test, batch_size=128)
print(model.metrics_names)
print(loss_NN)



print(hist.history.keys())
plt.plot(hist.history['loss'])

plt.rcParams["figure.figsize"] = (10,10)
plt.scatter(y_test,Y_pred)
plt.scatter(y_train,model.predict(x_train))

## Kernal KernelRidge
model_ridge = KernelRidge(kernel = 'rbf', gamma = 0.01, alpha = best_alpha).fit(x_train, y_train)
##plotting x,y as train, prediction. If perfect should be a line with slope of 1
y_pred = model_ridge.predict(x_train)
round(model_ridge.score(x_test, y_test) * 100, 2)

sns.set()
plt.rcParams["figure.figsize"] = (10,10)
plt.scatter(y_train,y_pred)

y_pred = model_ridge.predict(x_test)
plt.rcParams["figure.figsize"] = (10,10)
plt.scatter(y_test,y_pred)
print(y_test);
print(y_pred);

# somethings wrong with the neural net predictions and it's not giving binary predictions. not worth fixing
# print(Y_pred)
# #inverse one hot encoding
# pred = list()
# for i in range(len(Y_pred)):
#     pred.append(np.argmax(Y_pred[i]))
# print(pred)
#Converting one hot encoded test label to label
# test = list()
# for i in range(len(y_test)):
#     test.append(np.argmax(y_test[i])

np.sqrt(1.774617823490909)

coulomb_df_test = pd.read_pickle("coulomb_interactions_test_data.pic")
print(coulomb_df_test)
x_test_pca = pca.transform(coulomb_df_test)


#fit keras
neuraln = Sequential()
neuraln.add(Dense(128, input_dim=60, activation='relu'))
neuraln.add(Dense(256, kernel_initializer='normal',activation='relu'))
neuraln.add(Dense(256, kernel_initializer='normal',activation='relu'))
neuraln.add(Dense(256, kernel_initializer='normal',activation='relu'))
neuraln.add(Dense(1, kernel_initializer='normal', activation='linear'))
neuraln.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
neuraln.fit(x_pca, y, epochs=100, batch_size=10)
y_pred = neuraln.predict(x_test_pca)
y_pred = 10 ** y_pred #undo the y = np.log10(TI_df) used by the training
