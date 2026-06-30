#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import glob
import os
import re
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

def load_image_dataset(
        image_dir,
        labels_file,
        max_images=None,
        target_size=(256, 256),
        num_classes=5):

    # Load CSV labels
    labels_df = pd.read_csv(labels_file)
    labels_dict = dict(zip(labels_df['image'], labels_df['level']))

    # Get all images recursively
    all_files = glob.glob(os.path.join(image_dir, "**", "*.jpeg"), recursive=True)

    images = []
    labels = []
    count = 0

    for filepath in all_files:
        if max_images is not None and count >= max_images:
            break

        filename = os.path.basename(filepath)

        # Load image
        img = load_img(filepath, target_size=target_size)
        img_array = img_to_array(img) / 255.0

        # Match label
        base_name = re.sub(r'\s*\(.*\)', '', filename.split('.')[0])
        label = labels_dict.get(base_name)

        if label is not None:
            images.append(img_array)
            labels.append(label)
            count += 1

    # Convert to arrays
    images = np.array(images)
    labels = np.array(labels)

    # One-hot encode
    labels = to_categorical(labels, num_classes=num_classes)
    print("Loaded", images.shape[0], "images with labels")

    return images, labels


# In[2]:


def prepare_data(images, labels):
    labels_flat = np.argmax(labels, axis=1)
    images_flat = images.reshape((images.shape[0], -1))
    print(f'images shape {images_flat.shape}')
    print(f'labels shape {labels_flat.shape}')
    
    return images_flat, labels_flat




