from keras.preprocessing.image import ImageDataGenerator
from keras import losses
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

# import model from file
from model import CNN
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import cv2
import os

# Set number of epochs, learning rate, batch size and image dimensions
epochs = 2
init_learning = 1e-3
batch_size = 32
image_dims = (28, 28, 3)

# get the paths for the directory of images and set paths for output files
path = os.getcwd()
directory = path + "\\Dataset"
print(directory)
image_paths = sorted(list(paths.list_images(directory)))
model_path = path + "\\model.model"
label_path = path + "\\labels.pkl"
plot_path = path + "\\plot.png"
# shuffle the images  for training and test
random.seed(46)  # reproducible random seed for reusable results
random.shuffle(image_paths)

# begin loading images, pre process and get labels
data = []
labels = []

print("Loading data...")
for image_path in image_paths:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_dims[1], image_dims[0]))  # open cv flips the xy hence [1] [0]
    image = img_to_array(image)
    data.append(image)

    l = label = image_path.split(os.path.sep)[-2].split("_")
    # gets the folder name from the os path and uses it as a label
    labels.append(l)

# change the pixel intensity values to something keras can understand intensity/255 to get a 0-1 value
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# binarize labels
# Although a list of sets or tuples is a very intuitive format for multilabel data,
# it is unwieldy to process. This transformer converts between this intuitive format
# and the supported multilabel format: a (samples x classes) binary matrix indicating
# the presence of a class label.

mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i + 1, label))

# split the data into train and test
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=46)

# Generate batches of tensor image data with real-time data augmentation.
# The data will be looped over (in batches).
aug_data = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                              height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                              horizontal_flip=True, fill_mode="nearest")

# build the model from the CNN class in model.py
model = CNN.build(width=image_dims[1], height=image_dims[0], depth=image_dims[2],
                  classes=len(mlb.classes_), activation="sigmoid")

print("Loading model and compiling...")
# compile the model ready to fit
optimiser = Adam(lr=init_learning, decay=init_learning / epochs)
model.compile(loss="binary_crossentropy", optimizer=optimiser, metrics=["accuracy"])


print("begin training...")
# begin training
results = model.fit_generator(aug_data.flow(trainX, trainY, batch_size=batch_size), validation_data=(testX, testY),
                              steps_per_epoch=len(trainX) // batch_size, epochs=epochs, verbose=1)

print("training done! Saving model...")
# save trained model to file and labels
model.save(model_path)
f = open(label_path, "wb")
f.write(pickle.dumps(mlb))
f.close()

# generate a plot of training accuracy and loss and save to file
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), results.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), results.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), results.history["acc"], label="train_acc")
plt.plot(np.arange(0, epochs), results.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(plot_path)

