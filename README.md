# Densenet-Tensorflow
Tensorflow implementation of [Densenet](https://arxiv.org/abs/1608.06993) using **MNIST**
* The code that implements *this paper* is ***Densenet.py***
* The code that applied *dropout* is ***Densenet_dropout.py***
* There is a *slight difference*, I used ***AdamOptimizer***

## Requirements
* Tensorflow 1.x
* Python 3.x

## Results
* This result does ***not use dropout***
* The number of dense block layers is fixed to ***4***
* The highest test accuracy is ***99.2%*** (epoch = 50)

### Accuracy
![accuracy](./assests/acc.JPG)

### Loss
![Loss](./assests/loss.JPG)


## Author
Junho Kim
