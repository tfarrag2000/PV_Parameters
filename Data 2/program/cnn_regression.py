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

filters = [(16, 32), (16, 32, 64)]
no_dense_layers = [0, 1, 2]
dense_size = [16, 32, 64]
dropout = [0, 0.4, 0.6]
epo=1000
batch_size=8
for f in filters:
    for n_dense in no_dense_layers:
        for d_size in dense_size:
            for drop in dropout:
                Name = "Model8_filters_{}_dense_{}_denseSize_{}_Dropout_{}_{}".format(len(f), n_dense, d_size, drop, datetime.now().strftime("%Y%m%d-%H%M%S"))

                print("##############################################")
                print("   " + Name)
                print("##############################################")
                model = models.create_cnn(567, 756, 3, filters=f, no_dense_layers=n_dense, dense_size=d_size,
                                          dropout=drop, regress=True)

                print(model.summary())
                # opt = Adam(lr=1e-3, decay=1e-3 / 200)
                model.compile(loss="mean_squared_error", optimizer='adam' , metrics=['mse'])
                clbs = None
                earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2, mode='auto')
                mc = ModelCheckpoint('E:\\My Research Results\\Dr_Mosaad_Data2\\Models\\'+Name+'_best_model.h5', monitor='val_loss', mode='auto', verbose=1, save_best_only=True)
                # tensorboard
                logdir = "E:\\My Research Results\\Dr_Mosaad_Data2\\logs\\scalars\\" + Name
                tensorboard_callback = TensorBoard(log_dir=logdir)
                clbs = [earlyStopping, mc, tensorboard_callback]

                type(trainY)

                # train the model
                print("[INFO] training model...")
                history = model.fit(np.array(trainImagesX), np.array(trainY),
                                    validation_data=(np.array(testImagesX), np.array(testY)), epochs=epo, batch_size=8,
                                    verbose=2
                                    , callbacks=clbs)

                model.save('E:\\My Research Results\\Dr_Mosaad_Data2\\Models\\'+Name+'_Last_model.h5')
                # plot history loss
                pyplot.close()
                pyplot.plot(history.history['loss'], label='train_loss')
                pyplot.plot(history.history['val_loss'], label='test_loss')
                # pyplot.plot(history.history['val_mean_absolute_percentage_error'], label='MAPE')
                pyplot.legend()
                pyplot.savefig('E:\\My Research Results\\Dr_Mosaad_Data2\\experimentOutput\\' +Name+ "_loss_fig.png")
                pyplot.close()

                model = load_model('E:\\My Research Results\\Dr_Mosaad_Data2\\Models\\'+Name+'_best_model.h5')

                # make predictions on the testing data
                print("[INFO] predicting .............")
                preds = model.predict(testImagesX)
                # print(preds)
                print("[inverted] predicting .............")
                inv_preds = scaler.inverse_transform(preds)
                # print(inv_preds)
                print("original  .............")
                inv_original = scaler.inverse_transform(testY)
                # print(inv_original)

                # -----------------------------------------------------------------------------
                # print statistical figures of merit
                # -----------------------------------------------------------------------------

                import sklearn.metrics, math

                print("\n")
                print("Mean absolute error (MAE):      %f" %
                      sklearn.metrics.mean_absolute_error(inv_original, inv_preds))
                print("Mean squared error (MSE):       %f"
                      % sklearn.metrics.mean_squared_error(inv_original, inv_preds))
                print("Root mean squared error (RMSE): %f"
                      % math.sqrt(sklearn.metrics.mean_squared_error(inv_original, inv_preds)))
                print("R square (R^2):                 %f"
                      % sklearn.metrics.r2_score(inv_original, inv_preds))
                print(sklearn.metrics.mean_squared_error(inv_original, inv_preds, multioutput='raw_values'))
