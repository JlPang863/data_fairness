import jax
import argparse
# Options ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
args = parser.parse_args()
# data
args.dataset = 'celeba'
from jax.tree_util import tree_structure
print(tree_structure(args.__dict__))

def foo(**args):
    return args
args.dataset = 'celeba'
