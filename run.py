import sys
import debugpy
import argparse
import importlib

# sys.path.append('..')

argparser = argparse.ArgumentParser(
    usage="""python run.py {hmc,sgd,sgmcmc,vi} [--debug] [--params...]
example: python run.py hmc --debug --seed=0 --temperature=1. --dir=runs/hmc/mnist_subset160 --dataset_name=mnist --model_name=mlp_classification --step_size=3.e-5 --trajectory_len=1.5 --num_iterations=100 --subset_train_to=160"""
)
argparser.add_argument('model', help='Name of the model to train',
                       choices=['hmc', 'sgd', 'sgmcmc', 'vi'])
argparser.add_argument('--debug', action='store_true')

args, params = argparser.parse_known_args()

if args.debug:  # VSCode remote attach debugging
    debugpy.listen(5678)
    print('Waiting for VSCode debugger connection...')
    debugpy.wait_for_client()
    print('VSCode debugger connected.')

script_name = f'run_{args.model}'

# update argv to let the imported script parse arguments
sys.argv = [script_name+'.py', *params]

runner = importlib.import_module(script_name)
runner.run()
