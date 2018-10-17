#!/usr/bin/python3
from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

def main():
  california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

  california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))

  """ first 12K events as training """
  training_examples = preprocess_features(california_housing_dataframe.head(12000)) 
  training_targets = preprocess_targets(california_housing_dataframe.head(12000)) 
   
  """ last 5K events as test """
  validation_examples = preprocess_features(california_housing_dataframe.tail(5000)) 
  validation_targets = preprocess_targets(california_housing_dataframe.tail(5000)) 

  print("Training examples summary:")
  display.display(training_examples.describe())
  print("Validation examples summary:")
  display.display(validation_examples.describe())

  print("Training targets summary:")
  display.display(training_targets.describe())
  print("Validation targets summary:")
  display.display(validation_targets.describe())
  print(validation_targets)

  correlation_dataframe = training_examples.copy()

  """correlation matrix for the data"""
  print(correlation_dataframe.corr())

  linear_classifier = train_model(
    learning_rate=0.000003,
    steps=20000,
    batch_size=200,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets) 



def preprocess_features(california_housing_dataframe):
  """prepare inputs features"""
  """ we drop median_house_value"""

  selected_features = california_housing_dataframe[
    ["latitude",
     "longitude",
     "housing_median_age",
     "total_rooms",
     "total_bedrooms",
     "population",
     "households",
     "median_income"]]

  processed_features = selected_features.copy()
  """new feature rooms per person"""
  processed_features ["rooms_per_person"] = california_housing_dataframe ["total_rooms"]/california_housing_dataframe ["population"]

  return processed_features

##################################################

def preprocess_targets(california_housing_dataframe):
  """define median_house_value as target"""
  """ pd.DataFrame() empty dataset"""
  output_targets = pd.DataFrame() 
  """ different from tf4 astype = casting, in this case as float"""
  output_targets["median_house_value_is_high"] = (california_housing_dataframe["median_house_value"]> 265000).astype(float)
  print(output_targets)
  return output_targets

##################################################

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
  
  """convert data to npy array"""


  features = {key:np.array(value) for key,value in dict(features).items()} 

  ds = Dataset.from_tensor_slices((features,targets))
  ds = ds.batch(batch_size).repeat(num_epochs)

  if shuffle:
    ds = ds.shuffle(1000)

  features, labels = ds.make_one_shot_iterator().get_next()

  return features, labels 

##################################################

def train_model( learning_rate, steps, batch_size, training_examples, training_targets,
                 validation_examples, validation_targets):

  periods = 10
  steps_per_period = steps/periods
  
  #my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_classifier = tf.estimator.LinearClassifier(
      feature_columns=construct_feature_columns(training_examples),
      optimizer=my_optimizer)

  training_input_fn           = lambda: my_input_fn(training_examples, training_targets["median_house_value_is_high"], batch_size=batch_size)
  predict_training_input_fn   = lambda: my_input_fn(training_examples, training_targets["median_house_value_is_high"], num_epochs=1, shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, validation_targets["median_house_value_is_high"], num_epochs=1, shuffle=False)
 
  print(training_input_fn) 
  print("Training Model")
  print("LogLoss (on training data):")


  training_log_losses = []
  validation_log_losses = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_classifier.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.    
    training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
    training_probabilities = np.array([item['probabilities'] for item in training_probabilities])
    
    validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
    validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])
    
    training_log_loss = metrics.log_loss(training_targets, training_probabilities)
    validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, training_log_loss))
    # Add the loss metrics from this period to our list.
    training_log_losses.append(training_log_loss)
    validation_log_losses.append(validation_log_loss)
    evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)

    print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
    print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])
  print("Model training finished.")
  
  # Output a graph of loss metrics over periods.
  plt.ylabel("LogLoss")
  plt.xlabel("Periods")
  plt.title("LogLoss vs. Periods")
  plt.tight_layout()
  plt.plot(training_log_losses, label="training")
  plt.plot(validation_log_losses, label="validation")
  plt.legend()
  plt.show()

  validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
  # Get just the probabilities for the positive class.
  validation_probabilities = np.array([item['probabilities'][1] for item in validation_probabilities])

  false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
    validation_targets, validation_probabilities)
  plt.plot(false_positive_rate, true_positive_rate, label="our model")
  plt.plot([0, 1], [0, 1], label="random classifier")
  plt.legend(loc=2)
  plt.show()
  #plt.scatter(training_examples["latitude"], training_targets["median_house_value"])
  #plt.show()

  return linear_classifier

#################################################

def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Returns:
    A set of feature columns
  different from tf4 (bucket/binned dataset)
  """ 
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])

#################################################

if __name__== "__main__":
  main()



