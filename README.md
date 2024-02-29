Local graph smoothing for link prediction against unverisity attack
===============================================================================

About
-----
This project is the implementation of the paper "Local graph smoothing based on Unsupervised Learning for Link prediction".

This repo contains the codes, data and results reported in the paper.

Dependencies
-----
The script has been tested running under Python 3.7.7, with the following packages installed (along with their dependencies):

* networkx, scipy, sklearn, numpy, pickle


Some Python module dependencies are listed in `requirements.txt`, which can be easily installed with pip:

```
pip install -r requirements.txt
```


Usage: Model Training
-----
### Demo
Then a demo script is available by calling ```train.py```, as the following:

```
	python train.py --dataset cora 
```


Usage: Evaluation
-----
We provide the evaluation codes on the link prediction task here. 
We evaluate on three real-world datasets Cora, Citeseer and Polblogs. 


### Evaluate the test ASR
After finishing the training of the LGS-GNN, we then evaluate the test asr over the test nodes, as the following:

```
    python eval.py --dataset cora 
```

The verision of jupyter notebook is also supported as: eval.ipynb

