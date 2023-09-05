from argparse import ArgumentParser
from utils import load_config
parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config_ml1m.py')
parser.add_argument('--checkpoint', type=str)
args = parser.parse_args()
config = load_config(args.config)

