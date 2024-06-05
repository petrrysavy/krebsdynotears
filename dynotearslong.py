from dynotearstrain import train
import os

with open("krebsL.txt", "r") as file:
    lines = file.readlines()
files = ["krebsL" + os.sep + line.strip() for line in lines]

train(files, "long_graph.txt", "long.txt")
