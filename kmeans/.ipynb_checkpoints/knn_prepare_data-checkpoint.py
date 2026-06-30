#!/usr/bin/env python
# coding: utf-8

# In[ ]:

def prepare_data(images, labels):
    labels_flat = np.argmax(labels, axis=1)
    images_flat = images.reshape((images.shape[0], -1))
    print(f'images shape {images_flat.shape}')
    print(f'labels shape {labels_flat.shape}')

    return images_flat, labels_flat
