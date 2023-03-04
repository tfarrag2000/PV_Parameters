# import the necessary packages
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from pyimagesearch import datasets
from pyimagesearch import models
import numpy as np
import pandas as pd
from datetime import datetime
import locale
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot
import cv2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import CSVLogger

# construct the path to the input .txt file that contains information
# on each house in the dataset and then load the dataset
print("[INFO] loading attributes...")
df = datasets.load_attributes(r'D:\OneDrive\My Research\Dr_Mosaad\Data 2\Experiments_data.csv')

# scale our outputs to the range [0, 1]
names = df.columns
scaler = MinMaxScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=names)

# load the house images and then scale the pixel intensities to the
# range [0, 1]
print("[INFO] loading house images...")
images = datasets.load_images_from_folder(r'D:\OneDrive\My Research\Dr_Mosaad\Data 2\images')

images = images / 255.0

# print("---------------- image  --------------------")
# img = images[0]
# pyplot.imshow(img, cmap='gray', interpolation='bicubic')
# pyplot.xticks([]), pyplot.yticks([])  # to hide tick values on X and Y axis
# pyplot.show()
# #
# # exit()


# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
split = train_test_split(images, scaled_df, test_size=0.2, random_state=10)
(trainImagesX, testImagesX, trainY, testY) = split

print("---------------- Train Y  --------------------")
print(trainY)
print("---------------- Test Y  --------------------")
print(testY)


model = load_model('D:\\My Research Results\\Dr_Mosaad_Data2\\Models\\Model8_filters_2_dense_1_denseSize_32_Dropout_0.4_20191029-232731_resume_Training_best_model.h5')

# make predictions on the testing data
print("[INFO] predicting .............")
preds = model.predict(testImagesX)
print(preds)
print("[inverted] predicting .............")
inv_preds = scaler.inverse_transform(preds)
print(inv_preds)
print("original  .............")
inv_original = scaler.inverse_transform(testY)
print(inv_original)

# -----------------------------------------------------------------------------
# print statistical figures of merit
# -----------------------------------------------------------------------------

import sklearn.metrics, math

print("\n")
print(
    "Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(inv_original, inv_preds))
print(
    "Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(inv_original, inv_preds))
print("Root mean squared error (RMSE): %f" % math.sqrt(
    sklearn.metrics.mean_squared_error(inv_original, inv_preds)))
print("R square (R^2): %f" % sklearn.metrics.r2_score(inv_original, inv_preds))
print(sklearn.metrics.mean_squared_error(inv_original, inv_preds, multioutput='raw_values'))



