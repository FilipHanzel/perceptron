# Perceptron
Pure python implementation of a framework to build, train and evaluate simple (sequential) dense neural networks.

The point of this implementation is to build intuitive understanding of popular algorithms used for training neural networks.

## Dependencies
This implementation takes advantage of `tqdm` as training progress bar.
`matplotlib` is used for visualizations and `sklearn` for generating data in provided examples.

## Modules dependency structure
```
perceptron
├─ model - main class and all methods required to train the model
|   ├─ layer* - dense layer, only trainable layer implemented
|   |   └─ weight_init - methods of weight initialization
|   ├─ activation - activation functions/layers
|   ├─ dropout - regularization layer
|   ├─ decay - learning rate decays
|   ├─ loss - loss functions minimized during training
|   ├─ metric - metrics to get additional information of how model performs
|   ├─ optimizer - algorithms responsible for updating weights
|   └─ normalizer** - algorithms responsible for data normalization
└─ data_utils - data preprocessing tools and common functions used accross many modules
```
\* layer implements l1 and l2 regularization</br>
\*\* model has builtin input normalization capabilities, but normalizers can be easily used standalone

## Implemented algorithms
- layers
    - dense
- weight initialization
    - he
    - xavier
    - from gauss distribution
    - from uniform distribution
    - zeros
- activation
    - heavyside
    - linear
    - relu
    - leaky relu (parametrized relu)
    - sigmoid
    - tanh
    - softmax
- input normalization
    - min-max
    - zscore
- loss
    - mean square error
    - mean square logarithmic error
    - mean absolute error
    - binary crossentropy
    - categorical crossentropy
- learning rate decays
    - linear
    - polynomial
    - time based
    - exponential
    - step
- optimization
    - gradient descent
    - momentum
    - nesterov momentum
    - adagrad
    - rmsprop
    - adam
- regularization
    - l1 and l2 
    - dropout
- metrics
    - regression
        - mean absolute error
        - mean absolute percentage error
        - mean square error
        - root mean square error
        - cosine similarity
    - classification
        - binary accuracy
        - categorical accuracy
        - top k categorical accuracy

## Examples
- *xor* (binary classification) - simple example of building and training a model and decision boundary visualization
- [*sonar* (binary classification)](http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)) - attempt at solving sonar binary classification problem
- [*iris* (multiclass classification)](https://archive.ics.uci.edu/ml/datasets/iris) - attempt at solving iris multiclass classification problem
- [*auto-mpg* (regression)](https://www.kaggle.com/datasets/uciml/autompg-dataset) - attempt at solving mpg regression problem 
- *circles*  (binary classification) - dataset generated with sklearn, custom training loop with matplotlib interactive visualization 
- *activations* - visualization of activation functions
- *lr_decays* - visualization of learning rate decays
- *visualization_utils* - helper function to plot training history
