Local graph smoothing for link prediction against universal attack
===============================================================================


Due to recent security vulnerabilities identified in the dependencies of the code repository referenced in this paper, we have decided to temporarily restrict public access to the code. Specifically, the vulnerabilities include:

1. **Numpy**: Vulnerable versions are >= 1.9.0 and < 1.21. The recommended upgrade is to version 1.21 or later.
2. **Torch**: Vulnerable versions are <= 1.13.0. The recommended upgrade is to version 1.13.1 or later.
3. **TQDM**: Vulnerable versions are >= 4.4.0 and < 4.66.3. The recommended upgrade is to version 4.66.3 or later.

These vulnerabilities were highlighted in a security alert from GitHub for the week of May 21 - May 28. To ensure the security and integrity of the code, we are currently addressing these issues and will make the repository public again once the necessary updates and security reviews are completed.

If you are interested in obtaining a copy of my code and data, please feel free to contact me via email at [mangoding@cug.edu.cn]. I am more than willing to share my research findings with you and engage in further discussions regarding potential collaboration opportunities.

I sincerely appreciate your interest in my research and look forward to hearing from you.

mangoding


===============================================================================

About
-----
This project is the implementation of the paper "Local graph smoothing for link prediction against universal attack".

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

