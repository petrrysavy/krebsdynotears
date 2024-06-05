from dynotearstrain import train
import os

with open("krebsS.txt", "r") as file:
    lines = file.readlines()
files = ["krebsS" + os.sep + line.strip() for line in lines]

train(files, "short_graph.txt", "short.txt", measurements=range(1, 10000, 100))
