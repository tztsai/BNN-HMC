import argparse
from utils import script_utils
from utils import cmd_args_utils

parser = argparse.ArgumentParser()
cmd_args_utils.add_common_flags(parser)
args = parser.parse_args()

(train_set, test_set, net_apply, params, net_state, key, log_likelihood_fn,
 log_prior_fn, log_prior_diff_fn, predict_fn, ensemble_upd_fn, metrics_fns,
 tabulate_metrics) = script_utils.get_data_model_fns(args)

score = script_utils.evaluate(
    net_apply, params, net_state, train_set,
    test_set, predict_fn, metrics_fns, log_prior_fn)

print(score)