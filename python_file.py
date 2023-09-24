import random 
import os
import numpy as np
import cv2
import re

import PIL.Image as Image
import os

import matplotlib.pylab as plt
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import tensorflow as tf
import tensorflow_hub as hub

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def get_labels(annotation_path):
    # Open the annotation file and read the labels.
    with open(annotation_path, 'r') as annotation_file:
        labels = []
        for line in annotation_file:
            if line.strip().startswith('<name>'):
                object_label = line.strip().replace('<name>', '').replace('</name>', '')
                labels.append(object_label)

    # Return the list of labels.
    return labels


def filter_images(image_dir, annotation_dir, classes):
    image_paths = []
    for image_file in os.listdir(image_dir):
        base_name, extension = os.path.splitext(image_file)
        image_path = os.path.join(image_dir, image_file)
        annotation_path = os.path.join(annotation_dir, base_name + '.xml')
        # Open the annotation file and read the labels.
        labels_all = get_labels(annotation_path)

        # If the image contains any of the given classes, add it to the list of image paths.
        if any(label in classes for label in labels_all):
            image_paths.append(image_path)
            # image_paths.append(annotation_path)

    return image_paths

def separate_jpg_xml(files):
  jpg_files = []
  xml_files = []
  for file in files:
    if re.search(r'\.jpg$', file):
      jpg_files.append(file)
    elif re.search(r'\.xml$', file):
      xml_files.append(file)
  return jpg_files, xml_files



# Get the image and annotation directories.
image_dir = 'VOCdevkit/VOC2011/JPEGImages'
annotation_dir = 'VOCdevkit/VOC2011/Annotations'

# Choose the classes to filter the images by.
cat_A = ['car']
not_cat_A=[]
all_labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'cat', 'car' , 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

class_labels_not_A = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
# Filter the images.
path_cat_A = filter_images(image_dir, annotation_dir, cat_A)
# print(path_cat_A)


random_A_image_paths = random.sample(path_cat_A, int(len(path_cat_A) * random.uniform(0.1, 0.3)))
for classes in class_labels_not_A:
    temp =filter_images(image_dir, annotation_dir, classes)
    image_paths = random.sample(temp, int(len(temp) * (0.01)))
    not_cat_A=not_cat_A+image_paths
    
dataset = random_A_image_paths+not_cat_A

# labels_array = np.empty((len(image_dir), 0)).astype(object)
# for img_path in annotation_dir:
#     labels = get_labels(img_path)
#     for i in range(len(image_paths)):
#         labels_array[i] = labels



# image_dataset, annotation_dataset = separate_jpg_xml(dataset)
print(dataset)

# def get_labels(annotation_path):
#     # Open the annotation file and read the labels.
#     with open(annotation_path, 'r') as annotation_file:
#         labels = []
#         for line in annotation_file:
#             if line.strip().startswith('<name>'):
#                 object_label = line.strip().replace('<name>', '').replace('</name>', '')
#                 labels.append(object_label)

#     # Return the list of labels.
#     return labels
IMAGE_SHAPE = (224, 224)

model=tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=IMAGE_SHAPE+(3,))
])
# model = VGG16(weights='imagenet', include_top=True)  # include_top=True includes the final fully-connected layers

# Load and preprocess your images
image_paths = dataset
images = [image.load_img(img_path, target_size=(224, 224)) for img_path in image_paths]
images = [image.img_to_array(img) for img in images]
images = [preprocess_input(img) for img in images]

images = np.array(images)


features = model.predict(images)
feature_max_idx = np.empty(shape=(len(features)), dtype=np.int64)

for i in range(len(features)):
  predicted_label_index = np.argmax(features[i], axis=0)
  feature_max_idx[i] = predicted_label_index

print(feature_max_idx)

# predicted_label_index = np.argmax(features[1])
# print(predicted_label_index)
features[:5]
# print(features)
feature_matrix = tf.concat(features, axis=0)
# print(feature_matrix)

image_labels = []
with open("ImageNetLabels.txt", "r") as f:
  image_labels = f.read().splitlines()

image_labels[:5]
# image_labels[predicted_label_index]

lables_pred = [image_labels[i] for i in feature_max_idx]

print(lables_pred)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(features, lables_pred)
predicted_labels = knn.predict(features)
predicted_labels
accuracy = accuracy_score(lables_pred, predicted_labels)
print(accuracy)
