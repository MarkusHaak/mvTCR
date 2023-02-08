"""
python -u covid_optuna.py --model poe --split 0
"""
# comet-ml must be imported before torch and sklearn
import comet_ml

import sys
sys.path.append('..')

from tcr_embedding.models.model_selection import run_model_selection
import tcr_embedding.utils_training as utils
from tcr_embedding.utils_preprocessing import group_shuffle_split

import os
import argparse


utils.fix_seeds(42)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='poe')
parser.add_argument('--split', type=int)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--tcr_emb', type=str, default=None, help="obsm column in adata obj storing tcr embeddings")
parser.add_argument('--timeout', type=float, default=48., help='max optimization time in hours.')
parser.add_argument('--continue_study', action='store_true', help='continue optuna study if it already exists instead of overwriting it')
args = parser.parse_args()

if args.tcr_emb is not None:
    args.model += '_emb'

adata = utils.load_data('covid')

# subsample to get statistics
if args.split is not None:
    random_seed = args.split
    sub, non_sub = group_shuffle_split(adata, group_col='clonotype', val_split=0.2, random_seed=random_seed)
    train, val = group_shuffle_split(sub, group_col='clonotype', val_split=0.20, random_seed=random_seed)
    adata.obs['set'] = 'train'
    adata.obs.loc[non_sub.obs.index, 'set'] = '-'
    adata.obs.loc[val.obs.index, 'set'] = 'val'
adata = adata[adata.obs['set'].isin(['train', 'val'])]

study_name = f'Covid_{args.model}'
if args.tcr_emb:
    study_name += f'_tcr-emb_{args.tcr_emb}'
if args.split is not None:
    study_name += f"_split_{args.split}"

params_experiment = {
    'study_name': study_name,
    'comet_workspace': None,  # 'Covid',
    'model_name': args.model,
    'balanced_sampling': 'clonotype',
    'metadata': ['identifier', 'cell_type', 'condition', 'responsive', 'reactive_combined'],
    'save_path': os.path.join(os.path.dirname(__file__), '..', 'optuna', study_name),
    'tcr_emb': args.tcr_emb,
}
if args.model == 'rna':
    params_experiment['balanced_sampling'] = None

params_optimization = {
    'name': 'pseudo_metric',
    'prediction_labels': ['clonotype', 'cell_type'],
}

timeout = (args.timeout * 60 * 60) - 300
run_model_selection(adata, params_experiment, params_optimization, None, timeout, args.continue_study, args.gpus)
