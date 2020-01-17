import numpy as np
import pandas as pd
import pickle

# Open the files

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
print(TotalDataPoints)
################
TI = pd.read_pickle("TI.pic")
print(TI)
TI.min()
#higher transfer integral means easier jump. closer together.
#task, take the atoms and locations and predict the transfer integral

#get average position of atom
# x_positions = pd.DataFrame(dimer.columns[::2])
x_positions = dimer.iloc[:, ::3]
print(x_positions)
# Get xpositions of first half, i.e. the first monomer (A)
print(x_positions.iloc[:,0:30])
x_avg_1 = x_positions.iloc[:,0:30].sum(axis=1)/30
print(x_avg_1)
x_avg_2 = x_positions.iloc[:,30::].mean(axis=1)
print(x_positions.iloc[:,30::].sum(axis=1)/30)
print(x_positions.iloc[:,30::])
print(x_avg_2)

y_positions = dimer.iloc[:, 1::3]
print(y_positions.iloc[:,0:30])
y_avg_1 = y_positions.iloc[:,0:30].sum(axis=1)/30
print(y_avg_1)
y_avg_2 = y_positions.iloc[:,30::].mean(axis=1)
print(y_positions.iloc[:,30::]);


z_positions = dimer.iloc[:,2::3]
print(z_positions.iloc[:,0:30])
# z_avg_1 = z_positions.iloc[:,0:30].sum(axis=1)/30
z_avg_1 = z_positions.iloc[:,0:30].mean(axis=1)
print(z_avg_1)
z_avg_2 = z_positions.iloc[:,30::].mean(axis=1)
print(z_positions.iloc[:,30::])
print(z_avg_2)
# average_position = dimer.mean(axis=1)
# print(average_position)
# cols = ['xa','ya','za','xb','yb','zb']
# df = pd.DataFrame(columns=cols)
# av_pos_df = pd.DataFrame()
av_pos_df = pd.DataFrame({"xa":x_avg_1,"ya":y_avg_1,"za":z_avg_1,"xb":x_avg_2,"yb":y_avg_2,"zb":z_avg_2})
print(av_pos_df)
x_dist = abs(x_avg_1 - x_avg_2)
x_dist
y_dist = abs(y_avg_1 - y_avg_2)
y_dist;
z_dist = abs(z_avg_1 - z_avg_2)
z_dist;

dist_df = pd.DataFrame({"x":x_dist,"y":y_dist,"z":z_dist})
print(dist_df)

StandardScaler().fit_transform(dist_df)
