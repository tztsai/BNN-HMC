import sys
import debugpy
import argparse
import importlib

# sys.path.append('..')

argparser = argparse.ArgumentParser(
    usage="python run.py {hmc,sgd,sgmcmc,vi} [--debug] [script_params...]"
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
