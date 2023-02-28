# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os
import matplotlib.pyplot as plt


# Tamer
def load_attributes(inputPath):
    # initialize the list of column names in the CSV file and then
    # load it using Pandas
    cols = ["Case_Id", "L1", "L2", "distance", "C1", "C2", "Cm", "RL"]

    df = pd.read_csv(inputPath, sep=",", header=0, names=cols, index_col="Case_Id")
    # return the data frame
    selectedCols= [ "distance", "C2", "Cm", "RL"]
    return df.loc[:, selectedCols]


# Tamer
def process_attributes(df, train, test):
    # initialize the column names of the continuous data
    # continuous = ["L1", "L2", "distance", "C1", "C2", "Cm", "RL"]
    continuous = [ "distance", "C2",  "Cm", "RL"]
    # performin min-max scaling each continuous feature column to
    # the range [0, 1]
    cs = MinMaxScaler()
    train = cs.fit_transform(train[continuous])
    test = cs.transform(test[continuous])

    return (train, test)


# Tamer
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        print(filename)
        img = cv2.imread(os.path.join(folder, filename))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # thresh = 128
        # image to black and white
        # img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]

        # plt.imshow(img, cmap='gray', interpolation='bicubic')
        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # plt.show()

        images.append(img)

    return np.array(images)


# Tamer
def load_images(df, inputPath):
    # initialize our images array (i.e., the house images themselves)
    images = []

    # loop over the indexes of the houses
    for i in df.index.values:
        # find the four images for the house and sort the file paths,
        # ensuring the four are always in the *same order*
        basePath = os.path.sep.join([inputPath, "{}_*".format(i + 1)])
        housePaths = sorted(list(glob.glob(basePath)))

        # initialize our list of input images along with the output image
        # after *combining* the four input images
        inputImages = []
        outputImage = np.zeros((64, 64, 3), dtype="uint8")

        # loop over the input house paths
        for housePath in housePaths:
            # load the input image, resize it to be 32 32, and then
            # update the list of input images
            image = cv2.imread(housePath)
            image = cv2.resize(image, (32, 32))
            inputImages.append(image)

        # tile the four input images in the output image such the first
        # image goes in the top-right corner, the second image in the
        # top-left corner, the third image in the bottom-right corner,
        # and the final image in the bottom-left corner
        outputImage[0:32, 0:32] = inputImages[0]
        outputImage[0:32, 32:64] = inputImages[1]
        outputImage[32:64, 32:64] = inputImages[2]
        outputImage[32:64, 0:32] = inputImages[3]

        # add the tiled image to our set of images the network will be
        # trained on
        images.append(outputImage)

    # return our set of images
    return np.array(images)


# load_images_from_folder(r'C:\Users\DELL\Desktop\images')
# df = load_attributes(r'D:\OneDrive\My Research\Dr_Mosaad\Data 2\Experiments_data.csv')

# print(df)
