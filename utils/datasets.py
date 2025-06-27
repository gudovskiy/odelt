import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import jax


def get_dataset(dataset_name, batch_size, is_train, debug_overfit=False):
    print("Loading dataset", dataset_name)

    '''
    def deserialization_fn(data):
            image = data['image']
            image = tf.image.random_flip_left_right(image)
            image = tf.cast(image, tf.float32)
            image = image / 255.0
            image = (image - 0.5) / 0.5 # Normalize to [-1, 1]
            return image,  data['label']'''
    
    def deserialization_fn(data):
        image = data['image']
        image = tf.image.random_flip_left_right(image) if is_train else image
        min_side = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
        image = tf.image.resize_with_crop_or_pad(image, min_side, min_side)
        
        if dataset_name in ['imagenet256', 'lsunchurch']:
            image = tf.image.resize(image, (256, 256), antialias=True)
        elif dataset_name in ['imagenet64']:
            image = tf.image.resize(image, (64, 64), antialias=True)

        image = tf.cast(image, tf.float32) / 255.0 # [0:1]
        image = (image - 0.5) / 0.5 # Normalize to [-1, 1]
        return image, data['label']

    if dataset_name == 'imagenet256':
        #split = tfds.split_for_jax_process('train' if (is_train or debug_overfit) else 'validation', drop_remainder=True)
        split = 'train'
        name = 'imagenet2012'
    elif dataset_name == 'celebahq64':
        # split = tfds.split_for_jax_process('train' if is_train else 'validation', drop_remainder=True)
        split = 'train'
        name = 'celebahq64'
    elif dataset_name == 'celebahq256':
        # split = tfds.split_for_jax_process('train' if is_train else 'validation', drop_remainder=True)
        split = 'train'
        name = 'celebahq256'
    elif dataset_name == 'cifar10':
        # split = tfds.split_for_jax_process('train' if is_train else 'validation', drop_remainder=True)
        split = 'train'
        name = 'cifar10'
    elif dataset_name == 'lsunchurch':
        #split = tfds.split_for_jax_process('church-train' if is_train else 'church-test', drop_remainder=True)
        split = 'church-train'
        name = 'lsunc'
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    dataset = tfds.load(name, split=split)
    dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(10000, seed=42, reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = tfds.as_numpy(dataset)
    dataset = iter(dataset)
    return dataset