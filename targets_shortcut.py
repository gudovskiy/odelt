import jax
import jax.numpy as jnp
import numpy as np

def get_targets(FLAGS, key, train_state, images, labels, force_t=-1, force_dt=-1):
    base_key, label_key, bsttime_key, time_key, noise_key = jax.random.split(key, 5)
    info = {}

    # 1) =========== Sample dt. ============
    bootstrap_batchsize = FLAGS.batch_size // FLAGS.model['bootstrap_every']
    leftover_batchsize  = FLAGS.batch_size - bootstrap_batchsize
    BS = bootstrap_batchsize if not FLAGS.model['bootstrap_cfg'] else bootstrap_batchsize // 2
    
    log2_sections = np.log2(FLAGS.model['denoise_timesteps']).astype(np.int32)
    dt_base       = jax.random.randint(base_key, (BS,), minval=0, maxval=log2_sections)
    dt_sections   = jnp.power(2, dt_base) # [1, 2, 4, 8, 16, 32, 64]
    dt = 1 / dt_sections / 2 # [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128]

    # 2) =========== Sample t. ============
    t = jax.random.uniform(bsttime_key, (BS,))
    t = jnp.floor(dt_sections * t) / dt_sections
    t_full = t[:, None, None, None]

    # 3) =========== Generate Bootstrap Targets ============
    l_t = labels[:BS]
    x_1 = images[:BS]
    x_0 = jax.random.normal(noise_key, x_1.shape)
    v_t = x_1 - x_0
    x_t = v_t * t_full + x_0
    call_model_fn = train_state.call_model if FLAGS.model['bootstrap_ema'] == 0 else train_state.call_model_ema
    if not FLAGS.model['bootstrap_cfg']:
        v_b1 = call_model_fn(t, x_t, (dt_base+1, l_t), train=False)
        x_t2 = x_t + dt[:, None, None, None] * v_b1
        t2   =   t + dt
        v_b2 = call_model_fn(t2, x_t2, (dt_base+1, l_t), train=False)
        v_target = (v_b1 + v_b2) / 2
    else:
        x_t_cfg = jnp.concatenate([x_t, x_t], axis=0)
        t_cfg   = jnp.concatenate([t, t], axis=0)
        db_cfg  = jnp.concatenate([dt_base, dt_base], axis=0)
        dv_cfg  = jnp.concatenate([dt, dt], axis=0)
        l_cfg   = jnp.concatenate([l_t, jnp.ones(BS, dtype=jnp.int32) * FLAGS.model['num_classes']], axis=0)
        
        v_b1 = call_model_fn(t_cfg, x_t_cfg, (db_cfg+1, l_cfg), train=False)
        v_b1 = jnp.concatenate([v_b1[:BS], v_b1[BS:] + FLAGS.model['cfg_scale'] * (v_b1[:BS] - v_b1[BS:])], axis=0)
        
        t2_cfg   = t_cfg   + dv_cfg
        x_t2_cfg = x_t_cfg + dv_cfg[:, None, None, None] * v_b1
        v_b2 = call_model_fn(t2_cfg, x_t2_cfg, (db_cfg+1, l_cfg), train=False)
        v_b2 = jnp.concatenate([v_b2[:BS], v_b2[BS:] + FLAGS.model['cfg_scale'] * (v_b2[:BS] - v_b2[BS:])], axis=0)
        v_target = (v_b1 + v_b2) / 2
    
    bst_xt = x_t if not FLAGS.model['bootstrap_cfg'] else x_t_cfg
    bst_l  = l_t if not FLAGS.model['bootstrap_cfg'] else jnp.concatenate([l_t, l_t], axis=0)
    bst_t  =   t if not FLAGS.model['bootstrap_cfg'] else t_cfg
    bst_dt = dt_base if not FLAGS.model['bootstrap_cfg'] else db_cfg
    bst_vt = v_target

    # 4) =========== Generate Flow Matching Targets ============
    labels_dropout = jax.random.bernoulli(label_key, FLAGS.model['class_dropout_prob'], (leftover_batchsize,))
    l_t = jnp.where(labels_dropout, FLAGS.model['num_classes'], labels[:leftover_batchsize])
    info['dropped_ratio'] = jnp.mean(l_t == FLAGS.model['num_classes'])

    # Sample t.
    t = jax.random.uniform(time_key, (leftover_batchsize,))
    t = jnp.floor(FLAGS.model['denoise_timesteps'] * t) / FLAGS.model['denoise_timesteps']
    t_full = t[:, None, None, None] # [batch, 1, 1, 1]
    
    # Sample flow pairs x_t, v_t.
    x_1 = images[:leftover_batchsize]
    x_0 = jax.random.normal(noise_key, x_1.shape)
    v_t = x_1 - x_0
    x_t = v_t * t_full + x_0
    d_t = jnp.ones(leftover_batchsize, dtype=jnp.int32) * log2_sections

    # ==== 5) Merge Flow+Bootstrap ====
    x_t = jnp.concatenate([bst_xt, x_t], axis=0)
    t   = jnp.concatenate([bst_t,    t], axis=0)
    d_t = jnp.concatenate([bst_dt, d_t], axis=0)
    v_t = jnp.concatenate([bst_vt, v_t], axis=0)
    l_t = jnp.concatenate([bst_l,  l_t], axis=0)
    
    info['bootstrap_ratio'] = jnp.mean(d_t != log2_sections)
    info['v_magnitude_bootstrap'] = jnp.sqrt(jnp.mean(jnp.square(bst_vt)))
    info['v_magnitude_b1'] = jnp.sqrt(jnp.mean(jnp.square(v_b1)))
    info['v_magnitude_b2'] = jnp.sqrt(jnp.mean(jnp.square(v_b2)))

    return x_t, v_t, t, d_t, l_t, info