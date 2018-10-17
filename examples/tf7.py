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



  california_housing_test_data = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv", sep=",")
  california_housing_test_data = california_housing_test_data.reindex(np.random.permutation(california_housing_test_data.index))


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
  dnn_regressor = train_nn_regression_model(
    learning_rate=0.006,
    steps=3000,
    batch_size=100,
    hidden_units=[10, 10],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

  #for the test example it's enough to divide in features vs target 
  #predict the target given the features and compare the predicted target with the real target 
  test_examples = preprocess_features(california_housing_test_data)
  test_targets = preprocess_targets(california_housing_test_data)

  predict_testing_input_fn = lambda: my_input_fn(test_examples, 
                                               test_targets["median_house_value"], 
                                               num_epochs=1, 
                                               shuffle=False)

  test_predictions = dnn_regressor.predict(input_fn=predict_testing_input_fn)
  test_predictions = np.array([item['predictions'][0] for item in test_predictions])

  root_mean_squared_error = math.sqrt(
    metrics.mean_squared_error(test_predictions, test_targets))

  print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)




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
  # Scale the target to be in units of thousands of dollars.
  output_targets["median_house_value"] = (
    california_housing_dataframe["median_house_value"] / 1000.0)
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
def train_nn_regression_model( 
    learning_rate,
    steps,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):

  periods = 7
  steps_per_period = steps / periods

  # Create a linear classifier object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  dnn_regressor = tf.estimator.DNNRegressor(
      feature_columns=construct_feature_columns(training_examples),
      hidden_units=hidden_units,
      optimizer=my_optimizer,
  )

  # Create input functions.

  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["median_house_value"], 
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets["median_house_value"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets["median_house_value"], 
                                                    num_epochs=1, 
                                                    shuffle=False)
  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    dnn_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    
    validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    
    # Compute training and validation loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print("Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()
  plt.show()

  print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
  print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)

  return dnn_regressor
 
#################################################

def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """ 
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])

#################################################

def get_quantile_based_buckets(feature_values, num_buckets):
  quantiles = feature_values.quantile(
    [(i+1.)/(num_buckets + 1.) for i in range(num_buckets)])
  return [quantiles[q] for q in quantiles.keys()]

#################################################

def model_size(estimator):
  variables = estimator.get_variable_names()
  size = 0
  for variable in variables:
    if not any(x in variable 
               for x in ['global_step',
                         'centered_bias_weight',
                         'bias_weight',
                         'Ftrl']
              ):
      size += np.count_nonzero(estimator.get_variable_value(variable))
  return size

################################################
if __name__== "__main__":
  main()



