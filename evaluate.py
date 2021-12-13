import os
import argparse
import pandas as pd
from utils import script_utils
from utils import cmd_args_utils
from utils import checkpoint_utils

parser = argparse.ArgumentParser()
cmd_args_utils.add_common_flags(parser)
args, _ = parser.parse_known_args()
if _: print('Warning: unknown arguments:', _)

if args.debug:  # VSCode remote attach debugging
    import debugpy
    debugpy.listen(5678)
    print('Waiting for VSCode debugger connection...')
    debugpy.wait_for_client()
    print('VSCode debugger connected.')

# Initialize data, model, losses and metrics
(train_set, test_set, net_apply, params, net_state, key, log_likelihood_fn,
log_prior_fn, log_prior_diff_fn, predict_fn, ensemble_upd_fn, metrics_fns,
tabulate_metrics) = script_utils.get_data_model_fns(args)

def find(s): return s if s in args.dir else None
model_type = find('hmc') or find('sgmcmc') or find('sgd') or find('vi')
ckpt_type = model_type if model_type != 'vi' else 'sgd'

init_dict = script_utils.get_initialization_dict(args.dir, args, {})
ckpt_parser = getattr(checkpoint_utils, f'parse_{ckpt_type}_checkpoint_dict')
start_iteration, params, net_state, *_ = ckpt_parser(init_dict)

_, _, _, test_stats, train_stats = script_utils.evaluate(
    net_apply, params, net_state, train_set,
    test_set, predict_fn, metrics_fns, log_prior_fn)

print('\nTrain performance:')
print(pd.Series(train_stats))
print('\nTest performance:')
print(pd.Series(test_stats))
