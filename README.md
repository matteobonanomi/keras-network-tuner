# KERAS GRIDSEARCH PRE-BUILT CLASSES for Neural Networks Hyper Parameters Tuning

This repo aims at providing a simple and easy-to-use implementation of Scikit Learn grid search algorithm on Tensorflow 2.0 neural networks using Keras API.
Keras is by far the easiest way to build a neural network model, while scikit learn is IMHO the most complete machine learnig library. 
In particular, classic scikit algorithms can be easily optimized using built-in grid search functions (GridSearchCV, RandomizedSearchCV). 
At the moment of writing Keras and Tensorflow, as well as PyTorch, do not offer such an easy-to-use tuning methodology. 
Surfing on machine learning blogs and forums, I discovered a good way to achieve this objective, using KerasRegressor and KerasClassifier objects to build scikit-compliant neural networks.
I wrote some lines of code that I want to share on this repo, since many blogs can offer a sample spaghetti-code tutorial, while it's difficult to find a more robust code sample.
I developed a few Python classes to widen the hyper parameters than you can tune using scikit-learn function on a Keras neural network, 
incapsulating all the possible tweaks you would like to try (number of neurons per layer, dropouts, callbacks, ecc..) in a single and intuitive YAML configuration file.
Moreover you can use this Keras-based model or switch to any other scikit learn model at any time in your code, 
since they have compatible modules and attributes and they behave in the very same way when scikit's gridsearch is applied.