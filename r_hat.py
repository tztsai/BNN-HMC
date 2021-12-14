# %%
import os, re, sys
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from utils import script_utils
from utils import cmd_args_utils
from utils import checkpoint_utils

parser = argparse.ArgumentParser()
cmd_args_utils.add_common_flags(parser)
parser.add_argument("--dirs", type=str, nargs="+", required=True)
if '--dirs' not in sys.argv:
    sys.argv.extend("--dirs runs/covshift/hmc/gaussian/model_mlp_classification_wd_100.0_stepsize_1e-05_trajlen_0.15_burnin_10_mh_True_temp_1.0_seed_2/ runs/hmc/mnist/model_mlp_classification_wd_1.0_stepsize_3e-05_trajlen_1.5_burnin_10_mh_True_temp_1.0_seed_0/".split())
sys.argv.extend(['--dir', 'None'])
args, _ = parser.parse_known_args()
if _: print('Warning: unknown arguments:', _)

args.model_name = 'mlp_classification'
args.dataset_name = 'mnist'

if args.debug:  # VSCode remote attach debugging
    import debugpy
    debugpy.listen(5678)
    print('Waiting for VSCode debugger connection...')
    debugpy.wait_for_client()
    print('VSCode debugger connected.')

# %%
import tensorflow as tf
import tensorflow_probability as tfp

def parse_and_set_args(dir):
    *_, wd, _, stepsize, _, trajlen, _, burnin, _, mh, _, temp, \
        _, seed = dir.strip('/').split('_')
    args.dir = dir
    args.weight_decay = float(wd)
    args.temperature = float(temp)
    args.seed = int(seed)

parse_and_set_args(args.dirs[0])
(train_set, test_set, net_apply, params, net_state, key, log_likelihood_fn,
    log_prior_fn, log_prior_diff_fn, predict_fn, ensemble_upd_fn, metrics_fns,
    tabulate_metrics) = script_utils.get_data_model_fns(args)

def get_chain(dir):
    print(dir)
    parse_and_set_args(dir)
    
    ckpts = sorted([p for p in os.listdir(args.dir) if p.endswith(
        'pt')], key=lambda p: int(re.search('\d+', p)[0]))

    ckpt_parser = getattr(checkpoint_utils, f'parse_hmc_checkpoint_dict')
    w_chain = []
    f_chain = []

    def flatten(x):
        if hasattr(x, 'items'):
            return np.hstack([flatten(v) for v in x.values()])
        return x.reshape(-1)
    
    for ckpt in tqdm(ckpts):
        ckpt_dict = checkpoint_utils.load_checkpoint(os.path.join(args.dir, ckpt))
        _, params, net_state, *_ = ckpt_parser(ckpt_dict)

        net_state, test_predictions = predict_fn(net_apply, params, net_state, test_set)
        
        weights = flatten(params)
        preds = tf.nn.softmax(test_predictions[0])
        
        w_chain.append(weights)
        f_chain.append(preds)

    return w_chain, f_chain

def r_hat(chains):
    return tfp.mcmc.diagnostic.potential_scale_reduction(
        chains, independent_chain_ndims=1)

# %%
w_chains, f_chains = zip(*map(get_chain, args.dirs))
w_chains = np.swapaxes(np.asarray(w_chains), 0, 1)
f_chains = np.swapaxes(np.asarray(f_chains), 0, 1)
print('weights shape:', w_chains.shape)
print('preds shape:', f_chains.shape)

# %%
w_rhat = np.array(r_hat(w_chains)).reshape(-1)
f_rhat = np.array(r_hat(f_chains)).reshape(-1)

# %%
import matplotlib.pyplot as plt

plt.hist(w_rhat, log=True, bins=50)
plt.xlabel('$\^{R}$')
plt.title('Weight Space')
plt.grid(True)
plt.xlim(0.5, plt.xlim()[1])
plt.show()

# %%
plt.hist(f_rhat, log=True, bins=50)
plt.xlabel('$\^{R}$')
plt.title('Function Space')
plt.grid(True)
plt.xlim(0.8, plt.xlim()[1])
plt.show()