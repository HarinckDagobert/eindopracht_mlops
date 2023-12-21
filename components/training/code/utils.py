import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Activation, Dropout

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from typing import List


def getTargets(filepaths: List[str]) -> List[str]:
    labels = [fp.split('/')[-1].split('_')[0]
              for fp in filepaths]  # Get only the animal name

    return labels


def encodeLabels(y_train: List, y_test: List):
    label_encoder = LabelEncoder()
    y_train_labels = label_encoder.fit_transform(y_train)
    y_test_labels = label_encoder.transform(y_test)

    y_train_1h = to_categorical(y_train_labels)
    y_test_1h = to_categorical(y_test_labels)

    LABELS = label_encoder.classes_
    print(f"{LABELS} -- {label_encoder.transform(LABELS)}")

    return LABELS, y_train_1h, y_test_1h


def getFeatures(filepaths: List[str]) -> np.array:
    images = []
    for imagePath in filepaths:
        image = Image.open(imagePath).convert("L")  # Convert to grayscale
        image = np.array(image)
        # Add a third dimension to simulate the original RGB structure
        image = np.expand_dims(image, axis=-1)
        images.append(image)
    return np.array(images)


def buildModel(inputShape: tuple, classes: int) -> Sequential:
    model = Sequential()
    height, width, depth = inputShape
    inputShape = (height, width, depth)
    chanDim = -1
    model.add(Conv2D(32, kernel_size=(3, 3),
              activation='relu', input_shape=inputShape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))  # Value between 0 and 1
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))  # Value between 0 and 1
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.1))  # Value between 0 and 1
    model.add(Dense(3, activation='softmax'))

    return model


# Example usage for grayscale images
# model = buildModel((64, 64, 1), len(LABELS))
