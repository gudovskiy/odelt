import jax
import jax.experimental
import wandb, os
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial
from diffrax import diffeqsolve, ODETerm, SaveAt, ConstantStepSize, PIDController, Euler, Dopri5, Tsit5


def do_inference(
    FLAGS,
    train_state,
    step,
    dataset,
    dataset_valid,
    shard_data,
    vae_encode,
    vae_decode,
    update,
    get_fid_activations,
    imagenet_labels,
    visualize_labels,
    fid_from_stats,
    truth_fid_stats,
):
    with jax.spmd_mode('allow_all'):
        global_device_count = jax.device_count()
        key = jax.random.PRNGKey(42 + jax.process_index())
        batch_images, batch_labels = next(dataset)
        valid_images, valid_labels = next(dataset_valid)
        if FLAGS.model.use_stable_vae:
            batch_images = vae_encode(key, batch_images)
            valid_images = vae_encode(key, valid_images)
        batch_labels_sharded, valid_labels_sharded = shard_data(batch_labels, valid_labels)
        labels_uncond = shard_data(jnp.ones(batch_labels.shape, dtype=jnp.int32) * FLAGS.model['num_classes'])  # Null token
        eps = jax.random.normal(key, batch_images.shape)

        def process_img(img):
            img = img * 0.5 + 0.5
            img = jnp.clip(img, 0, 1)
            img = np.array(img)
            return img

        @partial(jax.jit, static_argnums=(5,))
        def call_model(train_state, images, t, dt, labels, use_ema=True):
            if use_ema and FLAGS.model.use_ema:
                call_fn = train_state.call_model_ema
            else:
                call_fn = train_state.call_model
            output = call_fn(t, images, (dt, labels), train=False)
            return output

        #################################################################
        @jax.jit
        def solve(train_state, images, t0, t1, y, use_ema=True, steps_or_tol=1):
            if use_ema and FLAGS.model.use_ema:
                call_fn = train_state.call_model_ema
            else:
                call_fn = train_state.call_model

            vf = ODETerm(call_fn)
            solver = Euler() if isinstance(steps_or_tol, int) else Dopri5()  # solver = Tsit5()
            controller = ConstantStepSize() if isinstance(steps_or_tol, int) else PIDController(rtol=steps_or_tol, atol=steps_or_tol)
            dt0 = 1.0/steps_or_tol if isinstance(steps_or_tol, int) else None
            dt = jnp.ones((images.shape[0], )) * np.log2(steps_or_tol) if isinstance(steps_or_tol, int) else jnp.zeros((images.shape[0], ))
            dt = dt.astype(jnp.int32)
            #saveat = SaveAt(ts=jnp.linspace(0, 1, FLAGS.inference_timesteps))
            sol = diffeqsolve(vf, solver, t0=t0, t1=t1, dt0=dt0, y0=images, args=(dt, y), stepsize_controller=controller)  #, saveat=saveat)'
            return sol
        #################################################################

        def do_fid_calc(cfg_scale, steps_or_tol):
            num_generations = FLAGS.inference_generations
            images_shape = batch_images.shape
            activations, x1, lab = [], [], []
            gens = tqdm(range(num_generations // FLAGS.batch_size))
            
            if isinstance(steps_or_tol, int):  # fixed-step Euler summation
                denoise_timesteps = steps_or_tol
                delta_t = 1.0 / denoise_timesteps    
                dt_base = (jnp.ones((images_shape[0], )) * np.log2(denoise_timesteps)).astype(jnp.int32)
                if FLAGS.model.train_type == 'livereflow' and denoise_timesteps < 128:
                    dt_base = jnp.zeros_like(dt_base)

                for gen in gens:
                    key = jax.random.PRNGKey(42)
                    key = jax.random.fold_in(key, gen)
                    key = jax.random.fold_in(key, jax.process_index())
                    eps_key, label_key = jax.random.split(key)
                    x = jax.random.normal(eps_key, images_shape)
                    labels = jax.random.randint(label_key, (images_shape[0],), 0, FLAGS.model.num_classes)
                    x, labels = shard_data(x, labels)

                    for ti in range(denoise_timesteps):
                        t = 1.0 * ti / denoise_timesteps  # From x_0 (noise) to x_1 (data)
                        t_vector = jnp.full((images_shape[0], ), t)
                        t_vector, dt_base = shard_data(t_vector, dt_base)
                        
                        if cfg_scale == 1:
                            v = call_model(train_state, x, t_vector, dt_base, labels)
                        elif cfg_scale == 0:
                            v = call_model(train_state, x, t_vector, dt_base, labels_uncond)
                        else:
                            v_pred_uncond = call_model(train_state, x, t_vector, dt_base, labels_uncond)
                            v_pred_label  = call_model(train_state, x, t_vector, dt_base, labels)
                            v = v_pred_uncond + cfg_scale * (v_pred_label - v_pred_uncond)

                        if FLAGS.model.train_type == 'consistency':
                            eps = shard_data(jax.random.normal(jax.random.fold_in(eps_key, ti), images_shape))
                            x1pred = x + v * (1-t)
                            x = x1pred * (t+delta_t) + eps * (1-t-delta_t)
                        else:
                            x = x + v * delta_t  # Euler sampling.
                    
                    x1.append(x)

            else:  # variable-step integration
                for gen in gens:
                    key = jax.random.PRNGKey(42)
                    key = jax.random.fold_in(key, gen)
                    key = jax.random.fold_in(key, jax.process_index())
                    eps_key, label_key = jax.random.split(key)
                    x = jax.random.normal(eps_key, images_shape)
                    labels = jax.random.randint(label_key, (images_shape[0],), 0, FLAGS.model.num_classes)
                    x, labels = shard_data(x, labels)

                    if cfg_scale == 1:
                        sol = solve(train_state, x, 0.0, 1.0, labels,        steps_or_tol=steps_or_tol)
                        x = sol.ys[-1]
                    elif cfg_scale == 0:
                        sol = solve(train_state, x, 0.0, 1.0, labels_uncond, steps_or_tol=steps_or_tol)
                        x = sol.ys[-1]
                    
                    x1.append(x)

            latency = gens.format_dict["elapsed"]/gens.format_dict["total"]
            
            for x in x1:
                if FLAGS.model.use_stable_vae:
                    x = vae_decode(x)  # Image is in [-1, 1] space.

                x = jax.image.resize(x, (x.shape[0], 299, 299, 3), method='bilinear', antialias=False)
                x = jnp.clip(x, -1, 1)
                acts = get_fid_activations(x)
                acts = jax.experimental.multihost_utils.process_allgather(acts)
                acts = np.array(acts)
                activations.append(acts)

            return activations, x1, lab, latency
        #################################################################

        def do_plots(inference_cfg, inference_t, inference_l):
            images_shape = batch_images.shape
            x1, lab = [], []
            # seeds & noise
            key = jax.random.PRNGKey(42)
            key = jax.random.fold_in(key, jax.process_index())
            eps_key, label_key = jax.random.split(key)
            x0 = jax.random.normal(eps_key, images_shape)
            labels = jax.random.randint(label_key, (images_shape[0],), 0, FLAGS.model.num_classes)
            x0, labels = shard_data(x0, labels)
            # fixed-step Euler
            for CFG,T,l in zip(inference_cfg, inference_t, inference_l):
                x = x0.copy()
                print('CFG:', CFG, 'T:', T, 'l:', l)
                delta_t = 1.0 / T
                d = jnp.ones((images_shape[0], )) * np.log2(T)
                for ti in range(T):
                    t = jnp.full((images_shape[0], ), ti / T)
                    l = jnp.full((images_shape[0], ), l)
                    t, d, l = shard_data(t, d, l)
                    if CFG == 1:
                        v = train_state.call_model_ema(t, x, (d, labels), train=False, blocks=l)
                    elif CFG == 0:
                        v = train_state.call_model_ema(t, x, (d, labels_uncond), train=False, blocks=l)
                    else:
                        v_pred_uncond = train_state.call_model_ema(t, x, (d, labels_uncond), train=False, blocks=l)
                        v_pred_label  = train_state.call_model_ema(t, x, (d, labels), train=False, blocks=l)
                        v = v_pred_uncond + CFG * (v_pred_label - v_pred_uncond)
                    
                    x = x + v * delta_t  # Euler sampling.
            
                if FLAGS.model.use_stable_vae:
                    x = vae_decode(x)  # Image is in [-1, 1] space.
                
                x1.append (np.array(jax.experimental.multihost_utils.process_allgather(x)))
                lab.append(np.array(jax.experimental.multihost_utils.process_allgather(labels)))

            return x1, lab
        #################################################################

        # generate results
        if jax.process_index() == 0:
            if FLAGS.fid_stats is not None:
                #steps_or_tol_list = [1, 2, 4, 8, 16, 32, 64, 128] if FLAGS.model['t_sampling'] == 'discrete' else [1, 4, 128, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
                if FLAGS.dataset_name == 'celebahq256':
                    cfg_scale_list    = 3*[0]
                    steps_or_tol_list = [1, 4, 128]
                elif FLAGS.dataset_name == 'imagenet256':
                    cfg_scale_list    = [1, 1, 1.5]
                    steps_or_tol_list = [1, 4,  64]
                for cfg_scale, steps_or_tol in zip(cfg_scale_list, steps_or_tol_list):
                    activations, x1, lab, latency = do_fid_calc(cfg_scale, steps_or_tol)
                    activations = np.concatenate(activations, axis=0)
                    activations = activations.reshape((-1, activations.shape[-1]))
                    mu1    = np.mean(activations, axis=0)
                    sigma1 = np.cov( activations, rowvar=False)
                    #print('our mu1:', mu1.shape, mu1)
                    #print('ref mu2:', truth_fid_stats['mu'].shape, truth_fid_stats['mu'])
                    fid_score = fid_from_stats(mu1, sigma1, truth_fid_stats['mu'], truth_fid_stats['sigma'])
                    print('FID: {:.2f} with {:.2f} per-batch latency, sec for {} steps/tol '.format(fid_score, latency, steps_or_tol))
                    wandb.log({f'fid/timesteps/{steps_or_tol}': fid_score}, step=0)
                    wandb.log({f'fid/latency/{steps_or_tol}': latency}, step=0)
            
            if FLAGS.inference_plot:
                inference_cfg = 3*[1.5, 1, 1]
                inference_t   = 3*[ 64, 4, 1]
                inference_l   = 3*[12] + 3*[8] + 3*[4]
                
                x1, lab = do_plots(inference_cfg, inference_t, inference_l)
                print('plot x1/lab:', len(x1), len(lab))
                imgs = np.concatenate(x1,  axis=0)
                labs = np.concatenate(lab, axis=0)
                print('plot imgs:', imgs.shape)
                print('plot labs:', labs.shape)
                np.save('{}_{}_imgs.npy'.format(FLAGS.dataset_name, FLAGS.model.depth), imgs)
                np.save('{}_{}_labs.npy'.format(FLAGS.dataset_name, FLAGS.model.depth), labs)
