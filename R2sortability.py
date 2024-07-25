import numpy as np
import pandas as pd
import os
import traceback
from utils import read_graph
from utils import path
from CausalDisco.analytics import r2_sortability

# build the Ground-Truth adjacency matrix
graph = read_graph(path + os.sep + "groundtruth.txt")
adjacency = np.zeros((16, 16))

labels = ["FUMARATE", "GTP", "H2O", "CIS-ACONITATE", "MALATE", "OXALOACETATE", "FAD", "SUCCINYL-COA", "NAD",
          "A-K-GLUTARATE", "GDP", "NADH", "CITRATE", "SUCCINATE", "ISOCITRATE", "ACETY-COA"]

for u, v in graph:
    uind = labels.index(u[:-5]) # strip the _lag
    vind = labels.index(v[:-5])
    adjacency[uind, vind] = 1

# prepare to evaluate the data - list of files
with open("krebsN.txt", "r") as file:
    lines = file.readlines()
seriesfiles = ["krebsN" + os.sep + line.strip() for line in lines]

threesfiles = ["krebs3" + os.sep + "threes"+str(i)+".tsv" for i in range(100)]

with open("krebsS.txt", "r") as file:
    lines = file.readlines()
shortfiles = ["krebsS" + os.sep + line.strip() for line in lines]

with open("krebsL.txt", "r") as file:
    lines = file.readlines()
longfiles = ["krebsL" + os.sep + line.strip() for line in lines]

files = [seriesfiles, threesfiles, shortfiles, longfiles]
names = ["krebsN", "krebs3", "krebsS", "krebsL"]

# evaluate the data
for filelist, name in zip(files, names):
    print("****************")
    print(name)

    data = [pd.read_table(path + os.sep + file, header=None, index_col=0).transpose() for file in filelist]
    R2sortability = np.zeros(len(data))
    # doublecheck that the order of labels is the same
    if (data[0].columns != labels).any():
        raise ValueError("Labels must be the same!")

    for i in range(len(data)):
        matrix = data[i].to_numpy()
        #print(matrix.shape)
        try:
            R2sortability[i] = r2_sortability(matrix, adjacency)
        except:
            # not sure why, but for krebsS and krebsL an error occurs for some reason
            #traceback.print_exc()
            R2sortability[i] = np.NaN

    print(R2sortability)
    print("Average: "+str(np.nanmean(R2sortability)))
    print("StDev: "+str(np.nanstd(R2sortability)))
    print("NaNs: "+str(np.count_nonzero(np.isnan(R2sortability))))

