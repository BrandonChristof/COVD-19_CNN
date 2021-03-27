'''
    Written by: Brandon Christof

    This script builds a multi-branch CNN model for classifying x-ray images of the chest
    as either Normal, Pneumonia, or COVID-19

    To execute:
        python CNN_XRAY.py
        python CNN_XRAY.py test
'''

import numpy as np
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Input, concatenate
import processing as p
import sys

(train1, train2, train_labels) = p.getData("trainFile.csv")
(test1, test2, test_labels) = p.getData("testFile.csv")

# Two CNN networks of the same structure (48-48-32-32)
# are combined into a single network that will classify the given inputs
def buildModel():
    # CNN 1 uses a cropped image without any further changes as its input
    cnn_1 = Input(shape=(p.imageSize, p.imageSize, 1))
    x = Conv2D(48, (3, 3), activation='relu')(cnn_1)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(48, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    output_1 = Flatten()(x)

    # CNN 2 uses a cropped image with image modifications (increase brightness and contrast) as its input
    cnn_2 = Input(shape=(p.imageSize, p.imageSize, 1))
    x = Conv2D(48, (3, 3), activation='relu')(cnn_2)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(48, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    output_2 = Flatten()(x)

    # Combines both CNNs into one for further learning
    concatenated = concatenate([output_1, output_2])
    x = Dense(48, activation='relu')(concatenated)
    x = Dense(16, activation='relu')(x)
    out = Dense(3, activation='softmax')(x)

    model = Model([cnn_1, cnn_2], out)
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model

# Uses Kera's fit method to train model for 8 epochs, and with a batch size of 8
# Every iteration is randomized by setting shuffle=True
def trainModel(model):
    model.fit([train1, train2], train_labels, shuffle=True, epochs=8, batch_size=8)
    saveModel(model)
    evaluateModel(model)

# Tests network for given model
def evaluateModel(model):
    model.evaluate([test1, test2], test_labels, batch_size=1)
    getConfusionMatrix(model)

# Saves network as json and h5 file
def saveModel(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")

# Loads pre-existing network within a json and h5 file
def loadModel():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    loaded_model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return loaded_model

# Prints confusion marix for trained network
def getConfusionMatrix(model):
    prediction = model.predict([test1, test2])
    covid = [0, 0, 0]
    pneumonia = [0, 0, 0]
    normal = [0, 0, 0]
    for i, p in enumerate(prediction):
        idx = p.tolist().index(max(p))
        t = test_labels[i].tolist().index(max(test_labels[i]))
        if idx == 0:
            normal[t] += 1
        elif idx == 1:
            pneumonia[t] += 1
        elif idx == 2:
            covid[t] += 1

    print("\nCONFUSION MATRIX")
    print(normal)
    print(pneumonia)
    print(covid)

if __name__ == "__main__":
    model = buildModel()
    trainModel(model)



