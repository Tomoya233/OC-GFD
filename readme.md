
# PyTorch Implementation of OC-GFD

This code provides a [PyTorch](https://pytorch.org/) implementation of the *OC-GFD* method proposed in our paper "Adaptive Gaussian Mixture Model with Hierarchical Message Propagation for One-Class Fraud Detection".

## Installation

This code is written in `Python 3.10` and requires `PyTorch 2.4.1`.

To run the code, we recommend using virtual environment like [conda](https://www.anaconda.com/download). we provide "requirements.txt" for installing dependencies.Run the following commands to set up environment for the code. 

```
conda create -n test python=3.10
conda activate test
pip install -r requirements.txt  
```

## Dataset
We use the YelpChi and Amazon datasets in the DGL library as datasets, which are automatically downloaded the first time they are used.


## Running experiments

After configuring the dependent environment, run

```
$ python main.py
```

and you will get results under the preset hyperparameter.

If you want to do further experiments, you can adjust the following hyperparameters in `parse.py` or add in the command line.

```
'--dataset': Dataset selection, yelp or amazon, default='yelp'
'--hidden_channels': Dimensions of the hidden layer, default=32
'--gmm_K': the number of GMM centers, default=3
'--lr': learning rate, default=0.001
'--dropout': default=0.2
```

We also provide script execution files for experimenting with different hyperparameters. run

```
$ run_script.sh
```
and you will get results under differrnt hyperparameters.


