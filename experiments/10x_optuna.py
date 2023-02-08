"""
python -u 10x_optuna.py --model poe --donor 1 --split 0
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
import config.constants_10x as const


utils.fix_seeds(42)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='poe')
parser.add_argument('--donor', type=str, default=None)
parser.add_argument('--filter_non_binder', choices=['all', 'val', 'False'], default='all')
parser.add_argument('--split', type=int)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--tcr_emb', type=str, default=None, help="obsm column in adata obj storing tcr embeddings")

parser.add_argument('--timeout', type=float, default=48., help='max optimization time in hours.')
parser.add_argument('--continue_study', action='store_true', help='continue optuna study if it already exists instead of overwriting it')
parser.add_argument('--knn_metric', default='weighted avg')
parser.add_argument('--normalize_binders', type=float, default=None)
args = parser.parse_args()

if args.tcr_emb is not None:
    args.model += '_emb'

adata = utils.load_data('10x')
if args.donor is not None:
    adata = adata[adata.obs['donor'] == f'donor_{args.donor}']
else:
    args.donor = 'donor_all'
if args.filter_non_binder == 'all':
    adata = adata[adata.obs['binding_name'].isin(const.HIGH_COUNT_ANTIGENS)]
elif args.filter_non_binder == 'val':
    # keep only most abundant antigen classes in validation set, but keep training set unchanged
    adata = adata[((adata.obs['set'] == 'val') & adata.obs['binding_name'].isin(const.HIGH_COUNT_ANTIGENS)) |\
                  (adata.obs['set'] != 'val')]

if args.tcr_emb:
    assert args.tcr_emb in adata.obsm , f"{args.tcr_emb} is not an obsm column in the adata object"


# subsample to get statistics
if args.split is not None:
    random_seed = args.split
    train_val, test = group_shuffle_split(adata, group_col='clonotype', val_split=0.20, random_seed=random_seed)
    train, val = group_shuffle_split(train_val, group_col='clonotype', val_split=0.25, random_seed=random_seed)

    adata.obs['set'] = 'train'
    adata.obs.loc[val.obs.index, 'set'] = 'val'
    adata.obs.loc[test.obs.index, 'set'] = 'test'
adata = adata[adata.obs['set'].isin(['train', 'val'])]

study_name = f'10x_{args.donor}_{args.model}_filtered_{args.filter_non_binder}'
if args.tcr_emb:
    study_name += f'_tcr-emb_{args.tcr_emb}'
if args.knn_metric != 'weighted avg':
    study_name += f"_metric_{args.knn_metric.replace(' ','-')}"
if args.split is not None:
    study_name += f"_split_{args.split}"
#if args.normalize_binders:
#    study_name += f"_normalize_{args.normalize_binders}"

params_experiment = {
    'study_name': study_name,
    'comet_workspace': None,
    'model_name': args.model,
    'balanced_sampling': 'clonotype',
    'metadata': ['binding_name', 'clonotype', 'donor'],
    'save_path': os.path.join(os.path.dirname(__file__), '..', 'optuna', study_name),
    'tcr_emb': args.tcr_emb,
}

if args.model == 'rna':
    params_experiment['balanced_sampling'] = None

params_optimization = {
    'name': 'knn_prediction',
    'prediction_column': 'binding_name',
    'knn_metric': args.knn_metric,
}

timeout = (args.timeout * 60 * 60) - 300
run_model_selection(adata, params_experiment, params_optimization, None, timeout, args.continue_study, args.gpus)
