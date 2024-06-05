from dynotearstrain import train
import os

files = ["krebs3" + os.sep + "threes"+str(i)+".tsv" for i in range(100)]

train(files, graph_file="threes_graph.txt", results_file="threes.txt")
