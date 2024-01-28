import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

"""
My attempts at importing keras from tensorflow is met with a well-documented "unresolved reference" error in both
PyCharm and in PyLance (VS Code). This issue has existed since 2019, and the provided work-around import statements
are used below. Please excuse the less-than pretty code required to do so.

You can also comment out the "work-around" imports, and try the original import statements I've commented-out below.
"""

# Import work-around for Keras -- have to directly reference the protected member Keras API.
from keras.api._v2 import keras

# Import work-around for Keras models library
Sequential = keras.models.Sequential

# Import work-around for Keras layers library
layers = keras.layers
Conv2D = layers.Conv2D
AveragePooling2D = layers.AveragePooling2D
Flatten = layers.Flatten
Dense = layers.Dense

# Import work-around for Keras metrics library
metrics = keras.metrics
Precision = metrics.Precision

# Import work-around for Keras optimizers library
optimizers = keras.optimizers
Adam = optimizers.Adam

# Import work-around for Keras utilities library
to_categorical = keras.utils.to_categorical

# Import work-around for Keras preprocessing library
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator


"""
The original import statements.
"""
# from tensorflow.keras.models import Sequential
# from layers import Conv2D, AveragePooling2D, Flatten, Dense
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.metrics import Precision


# The target folder containing the sample images, and the CSV file containing the image names with their labels,
# for easy manipulation while running experiments on different datasets/sub-datasets.
class_samples_folder = '10_classes'
csv_label_file = '10_class_clahe_cultivar_map.csv'

# Full Dataset Values -- Replace above to change which dataset is being tested.
# class_samples_folder = 'train_images'
# csv_label_file = 'train_cultivar_mapping.csv'

# Read the CSV file containing the image names and their respective class labels (i.e. cultivar name)
image_and_label_data = pd.read_csv(csv_label_file, header=None, names=['cultivar', 'image'])


# Preprocessing the data into arrays using numpy.
def preprocess_data(data, samples_folder):
    images = []
    labels = []
    for index, row in data.iterrows():
        image_file = row[0]  # Use the first column for image names
        image_path = os.path.join(samples_folder, image_file)
        sample = plt.imread(image_path)
        images.append(sample)
        labels.append(row[1])  # Use the second column for labels
    return np.array(images), np.array(labels)


# Divide the data into training and validation data (80% training, 20% validation)
# Random state is locked to an integer 1, to have the same result each time the model is run.
# Or at least, that's how I understood the explanation of the field.
train_data, val_data = train_test_split(image_and_label_data, test_size=0.2, random_state=1)

# Create the images and labels from the divided data for both the training and validation sets.
train_images, train_labels = preprocess_data(train_data, class_samples_folder)
val_images, val_labels = preprocess_data(val_data, class_samples_folder)

# Encode both the training and validation labels into integers.
encoder = LabelEncoder()
encoded_train_labels = encoder.fit_transform(train_labels)
encoded_val_labels = encoder.transform(val_labels)

# Perform "one-hot" encoding of the labels to represent the classes as binary vectors.
# The model requires the classes be binary vectors to use softmax for class prediction.
num_classes = 10   # change this to 10 if training the smaller dataset, 100 if training the full dataset.
train_labels = to_categorical(encoded_train_labels, num_classes)
val_labels = to_categorical(encoded_val_labels, num_classes)

# Data augmentation using flips, rotations, and zoom.
data_gen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    zoom_range=0.5,
)
data_gen.fit(train_images)


# LeNet Model Architecture
def create_lenet_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
    model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


# Image resolution/depth and the number of classes in the dataset, for easy manipulation while running experiments on
# different datasets/sub-datasets.
input_shape = (512, 512, 3)

# Create and compile the LeNet model, using ImageDataGenerator on the data and label arrays.
lenet_model = create_lenet_model(input_shape, num_classes)

# Though intuition would suggest that you can just add 'precision' to metrics, this is not the case.
# Thus, the clunky-looking work-around that references the keras.metrics.Precision library directly.
lenet_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy', Precision()])
history = lenet_model.fit(data_gen.flow(train_images, train_labels), validation_data=(val_images, val_labels), epochs=25)

# Make predictions for the class of each image in the validation set, and save those predictions to a CSV file.
val_predictions = lenet_model.predict(val_images)
val_predictions = np.argmax(val_predictions, axis=1)
val_prediction_labels = encoder.inverse_transform(val_predictions)

predictions = pd.DataFrame({'image': val_data.iloc[:, 0], 'cultivar': val_prediction_labels})
predictions.to_csv('predicted_classes.csv', index=False)


# Plotting all the metrics we've included in the model. In this case, the accuracy, loss, and precision for
# the training and validation sets.
def plot_metrics(history, metrics, save_path):
    figure, axis = plt.subplots(1, len(metrics), figsize=(15, 5))
    for i, metric in enumerate(metrics):
        axis[i].plot(history.history[metric], label=f'Training {metric}')
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            axis[i].plot(history.history[val_metric], label=f'Validation {metric}')
        axis[i].set_title(metric)
        axis[i].set_xlabel('Epoch')
        axis[i].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


metrics_to_plot = ['accuracy', 'loss', 'precision']
plot_save_path = 'training_validation_metrics.png'
plot_metrics(history, metrics_to_plot, plot_save_path)
