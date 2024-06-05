# krebsdynotears

This repository can be used to generate results in paper:
TODO: paper

To run the code, copy desired dataset from [this HuggingFace repository](https://huggingface.co/datasets/petrrysavy/krebs),
and the `groundtruth.txt` adjacency matrix into the `data` folder. Also, copy the list with the order of files into the root
of the repository. Then, use one of the `dynotearsnotrain.py`, `dynotearslong.py`, `dynotearsshort.py`, or `dynotearsthrees.py`
main files to run the code. To process the raw results into the plots, use `plots.py` file.

## Requirements
The repository requires the following packages to work:
 1. Pandas
 2. CausalNex
 3. Numpy
 4. Matplotlib
