'''
    Written by: Brandon Christof

    This script preproccesses the dataset by returning a full set of
    training images, training labels, testing images, and testing labels.
'''
from PIL import Image, ImageEnhance
import numpy as np
import csv
import random

location = "./Data/" # Folder where the datasets are
imageSize = 576      # How large the cropped image will be (576x576)
imageAdjust = 720    # What resolution to make the image before cropping

# Modifies image to be 20% brighter and 20% more contrasted
def modifyImage(img):
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.2)
    return img

# Resizes image to 720px while retaining aspect ratio
def resizeImage(img):
    global imageSize
    global imageAdjust

    w, h = img.size
    
    if w != imageAdjust and w <= h:
        basewidth = imageAdjust
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    elif h != imageAdjust and h <= w:
        baseheight = imageAdjust
        hpercent = (baseheight/float(img.size[1]))
        bsize = int((float(img.size[0])*float(hpercent)))
        img = img.resize((bsize,baseheight), Image.ANTIALIAS)

    w, h = img.size
    cropLeft = (w - imageSize)/2
    cropTop = (h - imageSize)/2
    cropRight = (w + imageSize)/2
    cropBottom = (h + imageSize)/2
    img = img.crop((cropLeft, cropTop, cropRight, cropBottom))
    
    return img

# Resizes, crops, and filters images
def processImages(img_norm, img_mod):
    img_norm = resizeImage(img_norm)
    img_mod = resizeImage(img_mod)
    img_mod = modifyImage(img_mod)
    img_norm = np.asarray(img_norm, dtype="int32")
    img_mod = np.asarray(img_mod, dtype="int32")
    img_norm = img_norm/255
    img_mod = img_mod/255  
    return (img_norm, img_mod)

# Returns 2D array for each image
def getImages(filename):
    img_norm = Image.open(location + "dataset/" + filename, 'r').convert("L")
    img_mod = Image.open(location + "dataset/" + filename, 'r').convert("L")
    (img1, img2) = processImages(img_norm, img_mod)
    img_norm.close()
    img_mod.close()
    return (img1, img2)

# Iterates through CSV and returns image information
def getMetaData(csvName):
    data = []
    name = location + csvName
    with open(name, newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            data.append(row)
    random.shuffle(data)
    csvfile.close()
    return data

# Creates and returns input numpy arrays for given dataset
def createDataset(name):
    meta = getMetaData(name)
    images_norm = []
    images_mod = []
    labels = []
    counter = 0
    size = len(meta)
    
    # Getting inputs defined by a csv file
    for m in meta:
        
        counter+=1
        (img1, img2) = getImages(m[1])

        images_norm.append(img1)
        images_mod.append(img2)

        if m[2] == "Normal":
            temp = np.array([1, 0, 0])
            labels.append(temp)
            print("NORMAL:    " + str(counter) + "/" + str(size))
        elif m[4] == "COVID-19":
            temp = np.array([0, 0, 1])
            labels.append(temp)
            print("COVID-19:  " + str(counter) + "/" + str(size))
        else:
            temp = np.array([0, 1, 0])
            labels.append(temp)
            print("PNEUMONIA: " + str(counter) + "/" + str(size))

    meta = None # clearing memory

    # Reshaping arrays to fit the CNN model
    labels = np.asarray(labels)
    images_norm = np.asarray(images_norm)
    images_mod = np.asarray(images_mod)
    images_norm = images_norm.reshape(len(images_norm), imageSize, imageSize, 1)
    images_mod = images_mod.reshape(len(images_mod), imageSize, imageSize, 1)
    
    return (images_norm, images_mod, labels)

# Main method used by the CNN script to gather input arrays and output labes
def getData(fileName):
    return createDataset(fileName)
