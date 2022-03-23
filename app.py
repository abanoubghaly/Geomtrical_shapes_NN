import numpy as np
import pandas as pd
import random

# image
from PIL import Image

# folder
import os
import glob

# visu
import matplotlib.pyplot as plt
plt.rc('image', cmap='gray')

# sklearn
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split

#tensorflow
# from tensorflow.keras import Sequential
# from tensorflow.keras import layers
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.utils import to_categorical
from keras.models import load_model
import sys


categories = ["circle", "square", "star", "triangle"]
im_width = 100
im_height = 100

# loading the model
loaded_model = load_model('trained_model')
np.set_printoptions(threshold=sys.maxsize)

while(True):
    prediction= None
    # loading and preprocessing the input image
    try:
        file = input("\nPath:")
        img = Image.open(file).convert('L')
        reshaped_input = np.array(img.resize((im_width, im_height)))
        reshaped_input = np.round((reshaped_input/255),3).copy()
        reshaped_input = reshaped_input.reshape((100,100,1))

        # running the model on the input
        prediction = loaded_model.predict(np.array([reshaped_input],))
    except:
        print("File not found or is not compatible")
        continue

    if 1 in np.round(prediction[0]):
        print("\n" + categories[list(np.round(prediction[0])).index(1)])
        fig = plt.figure(figsize=(10,7.5))
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        ax.axis('off')
        ax.set_title("Predicted: " + categories[list(np.round(prediction[0])).index(1)])
        ax.imshow(img)
        fig.suptitle("Geomtric shapes", fontsize=25, x=0.42)
        plt.show(block=True)
    else:
        print("No shape could be identified")