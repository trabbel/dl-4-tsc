# Deep Learning for WatchPlant Time Series Classification
This repository contains the code of the paper [Deep learning for time series classification: a review](https://link.springer.com/article/10.1007%2Fs10618-019-00619-1), originally published in [Data Mining and Knowledge Discovery](https://link.springer.com/journal/10618), also available on [ArXiv](https://arxiv.org/pdf/1809.04356.pdf). The code is slightly changed for the [WatchPlant project](https://watchplantproject.eu/).


## Data 
Create a folder in [archives](https://github.com/trabbel/dl-4-tsc/tree/master/archives) with subfolders for every individual experiment dataset. An experiment needs two files, named *experiment\_name\_TEST.tsv* and experiment\_name\_TRAIN.tsv. Test and train files should only include tab seperated values, first value per line should be the class as integer, all following values the features of the observation. 


## Code 
The code is divided as follows: 
* The [main.py](https://github.com/trabbel/dl-4-tsc/blob/master/main.py) python file contains the necessary code to run an experiement. 
* The [utils](https://github.com/trabbel/dl-4-tsc/tree/master/utils) folder contains the necessary functions to read the datasets and visualize the plots.
* The [classifiers](https://github.com/trabbel/dl-4-tsc/tree/master/classifiers) folder contains nine python files one for each deep neural network tested in our paper. 

To run a model on one dataset you should issue the following command: 
```
python3 main.py archive\_name experiment\_name classifier\_name iteration
```
To run all models for one dataset you should issue the following command:
```
python3 main.py all\_models archive\_name experiment\_name
```


## Prerequisites
All python packages needed are listed in [pip-requirements.txt](https://github.com/hfawaz/dl-4-tsc/blob/master/utils/pip-requirements.txt) file and can be installed simply using the pip command. 
The code now uses Tensorflow 2.0.

* [numpy](http://www.numpy.org/)  
* [pandas](https://pandas.pydata.org/)  
* [sklearn](http://scikit-learn.org/stable/)  
* [scipy](https://www.scipy.org/)  
* [matplotlib](https://matplotlib.org/)  
* [tensorflow-gpu](https://www.tensorflow.org/)  
* [keras](https://keras.io/)  
* [h5py](http://docs.h5py.org/en/latest/build.html)
* [keras_contrib](https://www.github.com/keras-team/keras-contrib.git)


## Reference

The code in this repository is based on [this paper](https://link.springer.com/article/10.1007%2Fs10618-019-00619-1), so please acknowledge the original authors if you reuse it:

```
@article{IsmailFawaz2018deep,
  Title                    = {Deep learning for time series classification: a review},
  Author                   = {Ismail Fawaz, Hassan and Forestier, Germain and Weber, Jonathan and Idoumghar, Lhassane and Muller, Pierre-Alain},
  journal                  = {Data Mining and Knowledge Discovery},
  Year                     = {2019},
  volume                   = {33},
  number                   = {4},
  pages                    = {917--963},
}
```
