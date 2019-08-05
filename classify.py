from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pickle
from imutils import paths
import cv2
import os

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
    kern = int((M+N)/10)
    print(M, N, Dim)
    temp = 0
    for i in range(0, M-kern, int(kern/4)):

        for j in range(0, N-kern, int(kern/10)):

            if i > temp:
                temp = i
                outputs[0].append(temp)
            else:
                outputs[0].append(temp)

            kernel = img[i:i+kern, j:j+kern]
            outputs[1].append(j)
            # cv2.imshow("k", kernel)
            # cv2.waitKey()
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
            print("[INFO] classifying image...")
            proba = model.predict(image)[0]
            idxs = np.argsort(proba)[::-1][:2]
            for (i, j) in enumerate(idxs):
                outputs[2].append(idxs[0])
                outputs[3].append(proba[j])

            # # show the probabilities for each of the individual labels
            # for (label, p) in zip(mlb.classes_, proba):
            #     print("{}: {:.2f}%".format(label, p * 100))

            # show the output image
            # cv2.imshow("Output", original)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

    print(outputs)








