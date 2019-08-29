from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pickle
from imutils import paths
import cv2
import os
from keras import models
from keras.preprocessing import image as kimage
import matplotlib.pyplot as plt
import PIL

path = os.getcwd()
model_path = path + "\\model.model"
label_path = path + "\\labels.pkl"
image_paths = sorted(list(paths.list_images(path+"\\images")))

with open(label_path, 'rb') as f:
    mlb = pickle.load(f)
f.close()

model = load_model(model_path)

for image_path in image_paths:

    outputs = [[], [], [], []]
    test = []
    original = img = cv2.imread(image_path)
    # cv2.imshow("image", image)
    # cv2.waitKey()
    M, N, Dim = np.shape(img)
    kern = 28  # int((M+N)/10)
    print(M, N, Dim)
    if int(M) < 28 or int(N) < 56:
        original = tmp = cv2.resize(img, (28, 28))
        img = tmp
    else:
        original = tmp = cv2.resize(img, (int(N/(M/28)), 28))
        img = tmp
    M, N, Dim = np.shape(img)
    print(M, N, Dim)
    temp = 0

    for i in range(0, N-(17)):
        j = 0
        outputs[0].append(i)

        kernel = img[j:j+kern, i:i+18]
        outputs[1].append(j)
        #cv2.imshow("k", kernel)
        #cv2.waitKey()
        # print(np.shape(test))
        # print(len(test))

        # for item in test:
        #     cv2.imshow("", item)
        #     cv2.waitKey()

        image = cv2.resize(kernel, (28, 28))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # classify the input image then find the indexes of the two class
        # labels with the *largest* probability
        probability = model.predict(image)[0]
        idxs = np.argsort(probability)[::-1][:1]
        for (i, j) in enumerate(idxs):
            outputs[2].append(str(idxs) + " " + str(mlb.classes_[j]))
            outputs[3].append((probability[j]*100))
        # proba = model.predict(image)[0]
        # idxs = np.argsort(proba)[::-1][:2]
        # for (i, j) in enumerate(idxs):
        #     outputs[2].append(idxs[0])
        #     outputs[3].append(proba[j])

        # # show the probabilities for each of the individual labels
        # for (label, p) in zip(mlb.classes_, proba):
        #     print("{}: {:.2f}%".format(label, p * 100))

        # show the output image
        # cv2.imshow("Output", original)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        # layer_name = 'my_layer'
        # intermediate_layer_model = model(inputs=model.input,
        #                                  outputs=model.get_layer(layer_name).output)
        # intermediate_output = intermediate_layer_model.predict(image)[0]
        # cv2.imshow("guesses", intermediate_output)
        # cv2.waitKey()

    print(outputs)
    prevchar = ""
    for i in range(0, len(outputs[0])):

        if outputs[3][i] > 90:
            currentchar = outputs[2][i]
            if currentchar != prevchar:
                prevchar = currentchar
                print("A {}% chance of character {} at x:{} , y:{}".format(str(outputs[3][i]), str(outputs[2][i]),
                                                                       str(outputs[0][i]), str(outputs[1][i])))

                cv2.rectangle(img, (outputs[0][i], outputs[1][i]),
                          (outputs[0][i]+kern, outputs[1][i]+kern), (255, 0, 0), 1)

    cv2.imshow("guesses", img)
    cv2.waitKey()

    img_path = 'C:\\Users\\Thoma\\Desktop\\Thesis\\Project\\dataset\\A\\A_0_track0022[17].png'
    img1 = cv2.imread(img_path)
    img1 = cv2.resize(img1, (28, 28))
    img_tensor = kimage.img_to_array(img1)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    layer_outputs = [layer.output for layer in model.layers[:12]]  # Extracts the outputs of the top 12 layers
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)  # Returns a list of five Numpy arrays: one array per layer activation
    print(activations)
    layer_names = []
    for layer in model.layers[:12]:
        layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                :, :,
                                col * images_per_row + row]
                channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,  # Displays the grid
                row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()


