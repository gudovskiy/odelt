import os
import tensorflow as tf
from typing import Any
import jax.numpy as jnp
from absl import app, flags
from functools import partial
import numpy as np
import tqdm
import jax
import jax.numpy as jnp
import flax
import optax
import wandb
from ml_collections import config_flags
import ml_collections

from utils.wandb import setup_wandb, default_wandb_config
from utils.train_state import TrainStateEma
from utils.checkpoint import Checkpoint
from utils.stable_vae import StableVAE
from utils.sharding import create_sharding, all_gather
from utils.datasets import get_dataset
from model import DiT
from helper_eval import eval_model
from helper_inference import do_inference

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', 'imagenet256', 'Dataset name: imagenet256/celebahq256')
flags.DEFINE_string('load_dir', None, 'Logging dir (if not None, save params)')
flags.DEFINE_string('save_dir', None, 'Logging dir (if not None, load params)')
flags.DEFINE_string('fid_stats', None, 'FID stats file')  # files 'celebahq256_fidstats_jax.npz, imagenet256_fidstats_jax.npz'
flags.DEFINE_integer('seed', 10, 'Random seed')  # Must be the same across all processes.
flags.DEFINE_integer('log_interval', 5000, 'Logging interval')
flags.DEFINE_integer('eval_interval', 20000, 'Eval interval')
flags.DEFINE_integer('save_interval', 10000, 'Eval interval')
flags.DEFINE_integer('batch_size', 16, 'Mini batch size')
flags.DEFINE_integer('max_steps', 100000, 'Number of training steps')
flags.DEFINE_integer('debug_overfit', 0, 'Debug overfitting')
flags.DEFINE_string('mode', 'train', 'Mode: train/eval')

flags.DEFINE_integer('inference_generations', 49152, 'Number of generations for inference.')
flags.DEFINE_integer('inference_plot', 0, 'Plot generated images.')

model_config = ml_collections.ConfigDict({
    'lr': 1e-4,
    'beta1': 0.9,
    'beta2': 0.999,
    'weight_decay': 0.1,
    'warmup': 5000,
    'dropout': 0.0,
    'hidden_size': 768, # 1152, # change this!
    'patch_size': 2,
    'num_heads': 12, # 16, # change this!
    'mlp_ratio': 4, # change this!
    'class_dropout_prob': 0.1,
    'num_classes': 1000, #1,
    'denoise_timesteps': 128,
    'cfg_scale': 0.0,
    'target_update_rate': 0.999,
    'use_ema': 1,
    'use_stable_vae': 1,
    'sharding': 'dp', # dp or fsdp.
    't_sampling': 'discrete', #'continuous', #'discrete',
    'dt_sampling': 'uniform',
    'bootstrap_cfg': 0,
    'bootstrap_every': 8, # Make sure its a divisor of batch size.
    'bootstrap_ema': 1,
    'bootstrap_dt_bias': 0,
    'train_type': 'shortcut', # or naive.
    'depth': 12, # 28, # change this!
    'depth_min': 4, # 12, # change this!
    'depth_group': 4, # change this!
    'depth_bootstrap': 8, # change this!
    'depth_wise': 0
})

wandb_config = default_wandb_config()

config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('model', model_config, lock_config=False)

##############################################
## Training Code.
##############################################
def main(_):
    tf.config.experimental.set_visible_devices([], "GPU")
    np.random.seed(FLAGS.seed)
    print("Using devices", jax.local_devices())
    device_count = len(jax.local_devices())
    global_device_count = jax.device_count()
    print("Device count", device_count)
    print("Global device count", global_device_count)
    local_batch_size = FLAGS.batch_size // (global_device_count // device_count)
    print("Global Batch: ", FLAGS.batch_size)
    print("Node Batch: ", local_batch_size)
    print("Device Batch:", local_batch_size // device_count)

    # Create wandb logger
    if jax.process_index() == 0:
        if FLAGS.mode == 'train':
            name = 'TRAIN_{}_D{}_G{}_{}'.format(FLAGS.model.train_type, FLAGS.model.depth_wise, FLAGS.model.depth_group, FLAGS.model.t_sampling)
            setup_wandb(FLAGS.model.to_dict(), project='shortcut_{}'.format(FLAGS.dataset_name), name=name, **FLAGS.wandb)
        else:
            name = 'EVAL_{}_D{}_G{}_{}'.format(FLAGS.model.train_type, FLAGS.model.depth_wise, FLAGS.model.depth_group, FLAGS.model.t_sampling)
            setup_wandb(FLAGS.model.to_dict(), project='eval_{}'.format(FLAGS.dataset_name), name=name, **FLAGS.wandb)
    
    dataset = get_dataset(FLAGS.dataset_name, local_batch_size, True, FLAGS.debug_overfit)
    dataset_valid = get_dataset(FLAGS.dataset_name, local_batch_size, False, FLAGS.debug_overfit)
    example_obs, example_labels = next(dataset)
    example_obs = example_obs[:1]
    example_obs_shape = example_obs.shape

    if FLAGS.model.use_stable_vae:
        vae = StableVAE.create()
        example_obs = vae.encode(jax.random.PRNGKey(0), example_obs)
        example_obs_shape = example_obs.shape
        vae_rng = jax.random.PRNGKey(42)
        vae_encode = jax.jit(vae.encode)
        vae_decode = jax.jit(vae.decode)
    else:
        vae_encode = None
        vae_decode = None
    
    if FLAGS.fid_stats is not None:
        from utils.fid import get_fid_network, fid_from_stats
        get_fid_activations = get_fid_network() 
        truth_fid_stats = np.load(FLAGS.fid_stats)
    else:
        from utils.fid import fid_from_stats
        get_fid_activations = None
        truth_fid_stats = None

    ###################################
    # Creating Model and put on devices.
    ###################################
    FLAGS.model.image_channels = example_obs_shape[-1]
    FLAGS.model.image_size = example_obs_shape[1]
    dit_args = {
        'patch_size': FLAGS.model['patch_size'],
        'hidden_size': FLAGS.model['hidden_size'],
        'num_heads': FLAGS.model['num_heads'],
        'mlp_ratio': FLAGS.model['mlp_ratio'],
        'out_channels': example_obs_shape[-1],
        'class_dropout_prob': FLAGS.model['class_dropout_prob'],
        'num_classes': FLAGS.model['num_classes'],
        'dropout': FLAGS.model['dropout'],
        'ignore_dt': False if (FLAGS.model['train_type'] in ('shortcut', 'livereflow')) else True,
        'depth': FLAGS.model['depth'],  # max depth in our method
        'depth_min': FLAGS.model['depth_min'],
        'depth_group': FLAGS.model['depth_group'],
        'depth_wise': FLAGS.model['depth_wise'],
    }
    model_def = DiT(**dit_args)
    tabulate_fn = flax.linen.tabulate(model_def, jax.random.PRNGKey(0))
    print(tabulate_fn(jnp.zeros((1,)), example_obs, (jnp.zeros((1,), dtype=jnp.int32), jnp.zeros((1,), dtype=jnp.int32))))

    warmup_linear = optax.warmup_constant_schedule(0.0, FLAGS.model['lr'], FLAGS.model['warmup'])
    cosine_decay = optax.cosine_decay_schedule(FLAGS.model['lr'], FLAGS.max_steps//10)
    lr_schedule = optax.schedules.join_schedules([warmup_linear, cosine_decay], [9*FLAGS.max_steps//10, FLAGS.max_steps//10])
    adam = optax.adamw(learning_rate=lr_schedule, b1=FLAGS.model['beta1'], b2=FLAGS.model['beta2'], weight_decay=FLAGS.model['weight_decay'])
    tx = optax.chain(adam)
    
    def init(rng):
        param_key, dropout_key, dropout2_key = jax.random.split(rng, 3)
        example_t = jnp.zeros((1,))
        example_dt = jnp.zeros((1,), dtype=jnp.int32)
        example_label = jnp.zeros((1,), dtype=jnp.int32)
        example_obs = jnp.zeros(example_obs_shape)
        model_rngs = {'params': param_key, 'label_dropout': dropout_key, 'dropout': dropout2_key}
        params = model_def.init(model_rngs, example_t, example_obs, (example_dt, example_label))['params']
        opt_state = tx.init(params)
        return TrainStateEma.create(model_def, params, rng=rng, tx=tx, opt_state=opt_state)
    
    rng = jax.random.PRNGKey(FLAGS.seed)
    train_state_shape = jax.eval_shape(init, rng)

    data_sharding, train_state_sharding, no_shard, shard_data, global_to_local = create_sharding(FLAGS.model.sharding, train_state_shape)
    train_state = jax.jit(init, out_shardings=train_state_sharding)(rng)
    jax.debug.visualize_array_sharding(train_state.params['FinalLayer_0']['Dense_0']['kernel'])
    #jax.debug.visualize_array_sharding(train_state.params['TimestepEmbedder_1']['Dense_0']['kernel'])
    #jax.experimental.multihost_utils.assert_equal(train_state.params['TimestepEmbedder_1']['Dense_0']['kernel'])
    start_step = 1

    print("Loading model from...", FLAGS.load_dir)
    if FLAGS.load_dir is not None:
        cp = Checkpoint(FLAGS.load_dir)
        replace_dict = cp.load_as_dict()
        print(replace_dict.keys())
        if 'train_state' in replace_dict:
            replace_dict = replace_dict['train_state']
        if 'opt_state' in replace_dict: del replace_dict['opt_state']
        if '_timestamp' in replace_dict: del replace_dict['_timestamp']
        
        print('\ntrain state before:', train_state.params.keys())
        print('\nload state before:', replace_dict['params'].keys())
        for k in train_state.params:
            if k in replace_dict['params']: train_state.params[k] = replace_dict['params'][k] # copy the parameters

        for k in train_state.params_ema:
            if k in replace_dict['params']: train_state.params_ema[k] = replace_dict['params'][k] # copy the parameters

        train_state.replace(step=replace_dict['step'])
        print('\ntrain state after:', train_state.params.keys())
        
        if FLAGS.wandb.run_id != "None": # If we are continuing a run.
            start_step = train_state.step
        train_state = jax.jit(lambda x : x, out_shardings=train_state_sharding)(train_state)
        print("Loaded model with step", train_state.step)
        train_state = train_state.replace(step=0)
        jax.debug.visualize_array_sharding(train_state.params['FinalLayer_0']['Dense_0']['kernel'])
        del cp

    if FLAGS.model.train_type == 'progressive' or FLAGS.model.train_type == 'consistency-distillation':
        train_state_teacher = jax.jit(lambda x : x, out_shardings=train_state_sharding)(train_state)
    else:
        train_state_teacher = None

    visualize_labels = example_labels
    visualize_labels = shard_data(visualize_labels)
    visualize_labels = jax.experimental.multihost_utils.process_allgather(visualize_labels)
    imagenet_labels = open('data/imagenet_labels.txt').read().splitlines()

    ###################################
    # Update Function
    ###################################

    @partial(jax.jit, out_shardings=(train_state_sharding, no_shard))
    def update(train_state, train_state_teacher, images, labels, force_t=-1, force_dt=-1):
        new_rng, targets_key, dropout_key, perm_key, b_t_key, bst_key = jax.random.split(train_state.rng, 6)
        info = {}
        id_perm = jax.random.permutation(perm_key, images.shape[0])
        images = images[id_perm]
        labels = labels[id_perm]
        images = jax.lax.with_sharding_constraint(images, data_sharding)
        labels = jax.lax.with_sharding_constraint(labels, data_sharding)

        if FLAGS.model['cfg_scale'] == 0: # For unconditional generation.
            labels = jnp.ones(labels.shape[0], dtype=jnp.int32) * FLAGS.model['num_classes']

        if 'fm' in FLAGS.model['train_type']:
            from baselines.targets_cfm import get_targets
            x_t, v_t, t, dt_base, labels, info = get_targets(FLAGS, targets_key, train_state, images, labels, force_t, force_dt)
        elif FLAGS.model['train_type'] == 'naive':
            from baselines.targets_naive import get_targets
            x_t, v_t, t, dt_base, labels, info = get_targets(FLAGS, targets_key, train_state, images, labels, force_t, force_dt)
        elif FLAGS.model['train_type'] == 'shortcut':
            from targets_shortcut import get_targets
            x_t, v_t, t, dt_base, labels, info = get_targets(FLAGS, targets_key, train_state, images, labels, force_t, force_dt)
        elif FLAGS.model['train_type'] == 'progressive':
            from baselines.targets_progressive import get_targets
            x_t, v_t, t, dt_base, labels, info = get_targets(FLAGS, targets_key, train_state, train_state_teacher, images, labels, force_t, force_dt)
        elif FLAGS.model['train_type'] == 'consistency-distillation':
            from baselines.targets_consistency_distillation import get_targets
            x_t, v_t, t, dt_base, labels, info = get_targets(FLAGS, targets_key, train_state, train_state_teacher, images, labels, force_t, force_dt)
        elif FLAGS.model['train_type'] == 'consistency':
            from baselines.targets_consistency_training import get_targets
            x_t, v_t, t, dt_base, labels, info = get_targets(FLAGS, targets_key, train_state, images, labels, force_t, force_dt)
        elif FLAGS.model['train_type'] == 'livereflow':
            from baselines.targets_livereflow import get_targets
            x_t, v_t, t, dt_base, labels, info = get_targets(FLAGS, targets_key, train_state, images, labels, force_t, force_dt)

        if FLAGS.model['depth_wise'] > 0:
            # 1) =========== Set Hyperparams ============
            group = FLAGS.model['depth_group']
            maxval = (FLAGS.model['depth'] - FLAGS.model['depth_min']) // group
            
            bst_size = FLAGS.batch_size // FLAGS.model['depth_bootstrap'] if FLAGS.model['depth_bootstrap'] > 0 else 0
            dat_size = FLAGS.batch_size - bst_size

            # 2) =========== Sample Blocks ============
            b_t = FLAGS.model['depth_min'] + group * jax.random.randint(b_t_key, (FLAGS.batch_size,), minval=0, maxval=maxval+1)
            if FLAGS.model['depth_bootstrap'] > 0:  # boostrapping length-wise consistency
                bst_b_t = FLAGS.model['depth_min'] + group * jax.random.randint(bst_key, (bst_size,        ), minval=0, maxval=maxval+1)
                # 3) =========== Generate Bootstrap Targets ============
                bst_x_t, bst_v_t, bst_t, bst_dt_base, bst_labels = x_t[dat_size:], v_t[dat_size:], t[dat_size:], dt_base[dat_size:], labels[dat_size:]
                call_model_fn = train_state.call_model if FLAGS.model['bootstrap_ema'] == 0 else train_state.call_model_ema
                tgt_v_t = call_model_fn(bst_t, bst_x_t, (bst_dt_base, bst_labels), train=False, blocks=bst_b_t)
                v_t = jnp.concatenate([v_t[:dat_size], tgt_v_t], axis=0)
                info['depth_mean'] = jnp.mean(1.0*b_t)
        else:
            b_t = None
        
        def loss_fn(grad_params):
            v_prime, logvars, activations = train_state.call_model(t, x_t, (dt_base, labels), train=True, rngs={'dropout': dropout_key}, params=grad_params, return_activations=True, blocks=b_t)
            mse_v = jnp.mean((v_prime - v_t) ** 2, axis=(1, 2, 3))
            loss = jnp.mean(mse_v)

            info = {
                'loss': loss,
                'v_magnitude_prime': jnp.sqrt(jnp.mean(jnp.square(v_prime))),
                **{'activations/' + k : jnp.sqrt(jnp.mean(jnp.square(v))) for k, v in activations.items()},
            }

            if FLAGS.model['train_type'] == 'shortcut' or FLAGS.model['train_type'] == 'livereflow':
                bootstrap_size = FLAGS.batch_size // FLAGS.model['bootstrap_every']
                info['loss_flow'] = jnp.mean(mse_v[bootstrap_size:])
                info['loss_bootstrap'] = jnp.mean(mse_v[:bootstrap_size])
            
            if FLAGS.model['depth_wise'] > 0:
                bst_size = FLAGS.batch_size // FLAGS.model['depth_bootstrap'] if FLAGS.model['depth_bootstrap'] > 0 else 0
                dat_size = FLAGS.batch_size - bst_size
                info['depth_consistency'] = jnp.mean(mse_v[dat_size:])

            return loss, info
        
        grads, new_info = jax.grad(loss_fn, has_aux=True)(train_state.params)
        info = {**info, **new_info}
        updates, new_opt_state = train_state.tx.update(grads, train_state.opt_state, train_state.params)
        new_params = optax.apply_updates(train_state.params, updates)

        info['grad_norm'] = optax.global_norm(grads)
        info['update_norm'] = optax.global_norm(updates)
        info['param_norm'] = optax.global_norm(new_params)
        info['lr'] = lr_schedule(train_state.step)

        train_state = train_state.replace(rng=new_rng, step=train_state.step + 1, params=new_params, opt_state=new_opt_state)
        train_state = train_state.update_ema(FLAGS.model['target_update_rate'])
        return train_state, info
    
    if FLAGS.mode != 'train':
        do_inference(FLAGS, train_state, None, dataset, dataset_valid, shard_data, vae_encode, vae_decode, update,
                       get_fid_activations, imagenet_labels, visualize_labels, fid_from_stats, truth_fid_stats)
        return

    ###################################
    # Train Loop
    ###################################

    for i in tqdm.tqdm(range(1 + start_step, FLAGS.max_steps + 1 + start_step), smoothing=0.1, dynamic_ncols=True):
        
        # Sample data.
        if not FLAGS.debug_overfit or i == 1:
            batch_images, batch_labels = shard_data(*next(dataset))
            if FLAGS.model.use_stable_vae:
                vae_rng, vae_key = jax.random.split(vae_rng)
                batch_images = vae_encode(vae_key, batch_images)

        # Train update.
        train_state, update_info = update(train_state, train_state_teacher, batch_images, batch_labels)

        if i % FLAGS.log_interval == 0 or i == 1:
            update_info = jax.device_get(update_info)
            update_info = jax.tree_map(lambda x: np.array(x), update_info)
            update_info = jax.tree_map(lambda x: x.mean(), update_info)
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}

            valid_images, valid_labels = shard_data(*next(dataset_valid))
            if FLAGS.model.use_stable_vae:
                valid_images = vae_encode(vae_rng, valid_images)
            _, valid_update_info = update(train_state, train_state_teacher, valid_images, valid_labels)
            valid_update_info = jax.device_get(valid_update_info)
            valid_update_info = jax.tree_map(lambda x: x.mean(), valid_update_info)
            train_metrics['training/loss_valid'] = valid_update_info['loss']

            if jax.process_index() == 0:
                wandb.log(train_metrics, step=i)

        if FLAGS.model['train_type'] == 'progressive':
            num_sections = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
            if i % (FLAGS.max_steps // num_sections) == 0:
                train_state_teacher = jax.jit(lambda x : x, out_shardings=train_state_sharding)(train_state)

        if i % FLAGS.eval_interval == 0:
            eval_model(FLAGS, train_state, train_state_teacher, i, dataset, dataset_valid, shard_data, vae_encode, vae_decode, update,
                       get_fid_activations, imagenet_labels, visualize_labels, fid_from_stats, truth_fid_stats)

        if i % FLAGS.save_interval == 0 and FLAGS.save_dir is not None:
            train_state_gather = jax.experimental.multihost_utils.process_allgather(train_state, tiled=True)
            if jax.process_index() == 0:
                filename = os.path.join(FLAGS.save_dir, str(i))
                cp = Checkpoint(filename, parallel=False)
                cp._values = train_state_gather.save()
                cp.save()
                del cp
            del train_state_gather

if __name__ == '__main__':
    app.run(main)