import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns
from sklearn.metrics import accuracy_scor
# import seaborn as sns; sns.set()
# Open the files

## Cosmetics for data readability
# sns.set(style='white', context='notebook', palette='deep')
# pylab.rcParams['figure.figsize'] = 18,12
# warnings.filterwarnings('ignore')
# mpl.style.use('ggplot')
# sns.set_style('white')
%matplotlib inline

atom_types = pd.read_pickle("atom_types.pic")
print(atom_types)
dimer = pd.read_pickle("dimer.pic")
print(dimer)
TI = pd.read_pickle("TI.pic")
# 10,000 dimers
# 180 points for locations.  60 atoms * 3 data points, xy,z, for each
dimer.shape
#################

TotalDataPoints = dimer.shape[0]
################
print(TI)
TI.min()
#higher transfer integral means easier jump. closer together.
#task, take the atoms and locations and predict the transfer integral

#get average position of atom
# x_positions = pd.DataFrame(dimer.columns[::2])
x_positions = dimer.iloc[:, ::3]
print(x_positions);
# Get xpositions of first half, i.e. the first monomer (A)
print(x_positions.iloc[:,0:30]);
x_avg_1 = x_positions.iloc[:,0:30].sum(axis=1)/30
print(x_avg_1);
x_avg_2 = x_positions.iloc[:,30::].mean(axis=1)
print(x_positions.iloc[:,30::].sum(axis=1)/30)
print(x_positions.iloc[:,30::])
print(x_avg_2)

y_positions = dimer.iloc[:, 1::3]
y_avg_1 = y_positions.iloc[:,0:30].sum(axis=1)/30
y_avg_2 = y_positions.iloc[:,30::].mean(axis=1)

z_positions = dimer.iloc[:,2::3]
z_avg_1 = z_positions.iloc[:,0:30].mean(axis=1)
z_avg_2 = z_positions.iloc[:,30::].mean(axis=1)
print(z_avg_2);

av_pos_df = pd.DataFrame({"xa":x_avg_1,"ya":y_avg_1,"za":z_avg_1,"xb":x_avg_2,"yb":y_avg_2,"zb":z_avg_2})
print(av_pos_df)
x_dist = abs(x_avg_1 - x_avg_2)
x_dist;
y_dist = abs(y_avg_1 - y_avg_2)
y_dist;
z_dist = abs(z_avg_1 - z_avg_2)
z_dist;

dist_df = pd.DataFrame({"x":x_dist,"y":y_dist,"z":z_dist})
print(dist_df)

dist_df = StandardScaler().fit_transform(dist_df)

x = dist_df
y = np.log10(TI)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

best_alpha = 0.001
# Now fit Ridge model
model_ridge = KernelRidge(kernel = 'rbf', gamma = 0.01, alpha = best_alpha).fit(x_train, y_train)

y_pred = model_ridge.predict(x_train)
sns.set()
plt.rcParams["figure.figsize"] = (10,10)
plt.scatter(y_train,y_pred)

y_pred = model_ridge.predict(x_test)
plt.rcParams["figure.figsize"] = (10,10)
plt.scatter(y_test,y_pred)
print(y_test)

print(y_pred)


acc_scorer = make_scorer(accuracy_score)

model_xgb = xgb.XGBClassifier(max_depth=6, n_estimators=300, learning_rate=0.05,colsample_bytree=.25).fit(x_train, y_train)
## Predictions ----------------
xgb_prediction = xgboost.predict(x_test)
xgb_score=accuracy_score(y_test, xgb_prediction)
print(xgb_score)
plt.scatter(y_test,xgb_prediction)
# xgb_score=accuracy_score(y_test, xgb_prediction)
# print(xgb_score)

x
# axes.set_xlim([xmin,xmax])
# axes.set_ylim([ymin,ymax])


#copy/paste of autohyperparameter
# # Choose some parameter combinations to try
# parameters = {'n_estimators': [4, 6, 9],
#               'max_features': ['log2', 'sqrt','auto'],
#               'criterion': ['entropy', 'gini'],
#               'max_depth': [2, 3, 5, 10],
#               'min_samples_split': [2, 3, 5],
#               'min_samples_leaf': [1,5,8]
#              }
#
# # Type of scoring used to compare parameter combinations
# acc_scorer = make_scorer(accuracy_score)
#
# # notes from me about grid search:
# # Grid search is the process of performing hyper parameter tuning in order to determine
# # the optimal values for a given model. This is significant as the performance of the entire
# # model is based on the hyper parameter values specified.
# # Run the grid search
# grid_obj = GridSearchCV(rfc, parameters, scoring=acc_scorer)
# grid_obj = grid_obj.fit(X_train, y_train)
#
# # Set the clf to the best combination of parameters
# rfc = grid_obj.best_estimator_
#
# # Fit the best algorithm to the data.
# rfc.fit(X_train, y_train)
