import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags

flags.DEFINE_float('sigma', 0.01, 'Noise STD in FM models.')

def get_targets(FLAGS, key, train_state, images, labels, force_t=-1, force_dt=-1):
    flow_type = FLAGS.model['train_type']
    dt_sampling = FLAGS.model['dt_sampling']
    sigma = FLAGS.sigma
    label_key, time_key, noise_key = jax.random.split(key, 3)
    info = {}

    labels_dropout = jax.random.bernoulli(label_key, FLAGS.model['class_dropout_prob'], (labels.shape[0],))
    labels_dropped = jnp.where(labels_dropout, FLAGS.model['num_classes'], labels)
    info['dropped_ratio'] = jnp.mean(labels_dropped == FLAGS.model['num_classes'])

    # Sample t.
    t = jax.random.uniform(time_key, (images.shape[0],))
    if FLAGS.model['t_sampling'] == 'discrete':
        t = jnp.round(FLAGS.model['denoise_timesteps'] * t) / FLAGS.model['denoise_timesteps']
    
    force_t_vec = jnp.ones(images.shape[0], dtype=jnp.float32) * force_t
    t = jnp.where(force_t_vec != -1, force_t_vec, t) # If force_t is not -1, then use force_t.
    t_full = t[:, None, None, None] # [batch, 1, 1, 1]
    
    # Sample flow pairs x_t, v_t.
    if 'latent' in FLAGS.dataset_name:
        x_1 = images[..., images.shape[-1] // 2:]
        x_0 = images[..., :images.shape[-1] // 2]
    else:
        x_1 = images
        x_0 = jax.random.normal(noise_key, images.shape)

    if flow_type == 'fm':
        mu_t = t_full * x_1
        sigma_t = 1 - (1 - sigma) * t_full
        x_t = mu_t + sigma_t * jax.random.normal(noise_key, images.shape)
        v_t = (x_1 - (1 - sigma) * x_t) / (1 - (1 - sigma) * t_full)
    elif flow_type == 'icfm' or flow_type == 'rfm':
        mu_t = (1 - t_full) * x_0 + t_full * x_1
        sigma_t = sigma if flow_type == 'icfm' else 0.0
        x_t = mu_t + sigma_t * jax.random.normal(noise_key, images.shape)
        v_t = x_1 - x_0
        
        # TODO: Add other flow types here.
        #elif flow_type == 'sifm':
        #elif flow_type == 'otcfm':

    dt_flow = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
    dt_base = jnp.ones(images.shape[0], dtype=jnp.int32) * dt_flow

    return x_t, v_t, t, dt_base, labels_dropped, info