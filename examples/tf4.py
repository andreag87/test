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

  correlation_dataframe = training_examples.copy()
  correlation_dataframe["target"] = training_targets["median_house_value"]

  """correlation matrix for the data"""
  print(correlation_dataframe.corr())

  #minimal_features = ["latitude", "longitude","rooms_per_person","median_income"]
 
  #assert minimal_features, "You must select at least one feature!"

  #minimal_training_examples = training_examples[minimal_features]
  #minimal_validation_examples = validation_examples[minimal_features]
  
 # selected_training_examples = select_and_transform_features(training_examples)
 # selected_validation_examples = select_and_transform_features(validation_examples)

#  print(selected_training_examples)


  _ = train_model(
    learning_rate=1.0,
    steps=500,
    batch_size=100,
    feature_columns=construct_feature_columns(training_examples),
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
  output_targets["median_house_value"] = (california_housing_dataframe["median_house_value"]/1000)
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

def train_model( learning_rate, steps, batch_size, feature_columns, training_examples, training_targets,
                 validation_examples, validation_targets):

  periods = 10
  steps_per_period = steps/periods
  
  my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  
  linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns, optimizer = my_optimizer)

  training_input_fn       = lambda : my_input_fn(training_examples, training_targets["median_house_value"], batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, training_targets["median_house_value"], num_epochs=1, shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, validation_targets["median_house_value"], num_epochs=1, shuffle=False)
  
  print("Training Model")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []

  for period in range (0, periods):
    linear_regressor.train(input_fn=training_input_fn, steps=steps_per_period)
    

    training_predictions   = linear_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])

    validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
  
    # print training and validation loss
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
 
    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
 
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
  #plt.scatter(training_examples["latitude"], training_targets["median_house_value"])
  #plt.show()

  return linear_regressor

##################################################

def select_and_transform_features(source_df):
  LATITUDE_RANGES = zip(range(32, 44), range(33, 45))
  print(LATITUDE_RANGES)
  selected_examples = pd.DataFrame()
  selected_examples["median_income"] = source_df["median_income"]
  for r in LATITUDE_RANGES:
    selected_examples["latitude_%d_to_%d" % r] = source_df["latitude"].apply(lambda l: 1.0 if l >= r[0] and l < r[1] else 0.0)
  return selected_examples

##################################################

def get_quantile_based_boundaries(feature_values, num_buckets):
  boundaries = np.arange(1.0, num_buckets) / num_buckets
  quantiles = feature_values.quantile(boundaries)
  return [quantiles[q] for q in quantiles.keys()]

#################################################

def construct_feature_columns(training_examples):
  """Construct the TensorFlow Feature Columns.

  Returns:
    A set of feature columns
  """ 

  households = tf.feature_column.numeric_column("households")
  longitude = tf.feature_column.numeric_column("longitude")
  latitude = tf.feature_column.numeric_column("latitude")
  housing_median_age = tf.feature_column.numeric_column("housing_median_age")
  median_income = tf.feature_column.numeric_column("median_income")
  rooms_per_person = tf.feature_column.numeric_column("rooms_per_person")
  
  # Divide households into 7 buckets.
  bucketized_households = tf.feature_column.bucketized_column(
    households, boundaries=get_quantile_based_boundaries(
      training_examples["households"], 7))

  print(training_examples)
  # Divide longitude into 10 buckets.
  bucketized_longitude = tf.feature_column.bucketized_column(
    longitude, boundaries=get_quantile_based_boundaries(
      training_examples["longitude"], 10))

  #
  # YOUR CODE HERE: bucketize the following columns, following the example above:
  #
  bucketized_latitude = tf.feature_column.bucketized_column(
    latitude, boundaries=get_quantile_based_boundaries(
      training_examples["latitude"], 10))

  bucketized_housing_median_age = tf.feature_column.bucketized_column(
    housing_median_age, boundaries=get_quantile_based_boundaries(
      training_examples["housing_median_age"], 7))

  bucketized_median_income = tf.feature_column.bucketized_column(
    median_income, boundaries=get_quantile_based_boundaries(
      training_examples["median_income"], 7))

  bucketized_rooms_per_person = tf.feature_column.bucketized_column(
    rooms_per_person, boundaries=get_quantile_based_boundaries(
      training_examples["rooms_per_person"], 7))

  long_x_lat = tf.feature_column.crossed_column(
    set([bucketized_longitude, bucketized_latitude]), hash_bucket_size=1000) 

  income_x_lat = tf.feature_column.crossed_column(
    set([bucketized_median_income, bucketized_latitude]), hash_bucket_size=1000) 

  feature_columns = set([
    bucketized_longitude,
    bucketized_latitude,
    bucketized_housing_median_age,
    bucketized_households,
    bucketized_median_income,
    bucketized_rooms_per_person, long_x_lat,
    income_x_lat])
  
  return feature_columns

#################################################

if __name__== "__main__":
  main()



