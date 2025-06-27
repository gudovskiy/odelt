import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from utils.fid import get_fid_network, fid_from_stats
get_activations = get_fid_network()

dataset_name = 'imagenet2012'
#dataset_name = 'celebahq64'
#dataset_name = 'celebahq256'
#dataset_name = 'cifar10'

def deserialization_fn(data):
    image = data['image']
    min_side = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
    image = tf.image.resize_with_crop_or_pad(image, min_side, min_side)

    if dataset_name in ['imagenet2012', 'lsunchurch']:
        image = tf.image.resize(image, (256, 256), antialias=True)
    elif dataset_name == 'celebahq64':
        image = tf.image.resize(image, (64, 64), antialias=True)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    image = tf.cast(image, tf.float32) / 255.0
    image = (image - 0.5) / 0.5 # Normalize to [-1, 1]
    return image

batch_size = 2048
split = tfds.split_for_jax_process('train', drop_remainder=True)
dataset = tfds.load(dataset_name, split=split)
dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(10000, seed=42, reshuffle_each_iteration=True)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
dataset = tfds.as_numpy(dataset)
dataset = iter(dataset)

activations = []
for i, batch in enumerate(dataset):
    print('{:d} ({:f})'.format(i, (len(activations)*batch_size) / 1_200_000.0))
    
    # For the last batch, we need to pad with zeros
    if batch.shape[0] < batch_size:
        zeros_added = batch_size - batch.shape[0]
        batch = np.concatenate([batch, np.zeros((batch_size - batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3]))], axis=0)
    else:
        zeros_added = 0

    batch = jnp.array(batch)
    #batch = batch.reshape((len(jax.local_devices()), -1, *batch.shape[1:])) # [devices, batch//devices, etc..]
    #print(batch.shape, jnp.min(batch), jnp.max(batch))
    batch = jax.image.resize(batch, (batch.shape[0], 299, 299, 3), method='bilinear', antialias=False)
    batch = jnp.clip(batch, -1, 1)
    
    preds = get_activations(batch)
    preds = preds.reshape((batch_size, -1))
    preds = np.array(preds)
    if zeros_added > 0:
        preds = preds[:-zeros_added]
    activations.append(preds)
activations = np.concatenate(activations, axis=0)
mu1 = np.mean(activations, axis=0)
sigma1 = np.cov(activations, rowvar=False)
np.savez('data/{}_fidstats_jax.npz'.format(dataset_name), mu=mu1, sigma=sigma1)
