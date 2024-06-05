from dynotearstrain import train
import os

with open("krebsN.txt", "r") as file:
    lines = file.readlines()
files = ["krebsN" + os.sep + line.strip() for line in lines]

train(files, "series_graph.txt", "results.txt")
