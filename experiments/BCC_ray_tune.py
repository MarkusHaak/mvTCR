# python -W ignore 10x_ray_tune.py --model bigru --name test
from comet_ml import Experiment
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import scanpy as sc
import os
import pickle
from datetime import datetime
import argparse
import importlib
import sys
sys.path.append('../config_tune')

import ray
from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.optuna import OptunaSearch


def trial_dirname_creator(trial):
	return f'{datetime.now().strftime("%Y%m%d_%H-%M-%S")}_{trial.trial_id}'


def correct_params(params):
	"""
	Ray Tune can't sample within lists, so this helper function puts the values back into lists
	:param params: hyperparameter dict
	:return: corrected hyperparameter dict
	"""
	params['loss_weights'] = [params['loss_weights_scRNA'], params['loss_weights_seq'], params['loss_weights_kl']]
	params['shared_hidden'] = [params['shared_hidden']]
	if 'num_layers' in params:
		params['shared_hidden'] = params['shared_hidden'] * params['num_layers']

	if 'loss_scRNA' in params:
		params['losses'][0] = params['loss_scRNA']

	if 'gene_hidden' in params['scRNA_model_hyperparams']:
		params['scRNA_model_hyperparams']['gene_hidden'] = [params['scRNA_model_hyperparams']['gene_hidden']]

	if 'num_layers' in params['scRNA_model_hyperparams']:
		params['scRNA_model_hyperparams']['gene_hidden'] = params['scRNA_model_hyperparams']['gene_hidden'] * params['scRNA_model_hyperparams']['num_layers']

	if params['seq_model_arch'] == 'CNN':
		params['seq_model_hyperparams']['num_features'] = [
			params['seq_model_hyperparams']['num_features_1'],
			params['seq_model_hyperparams']['num_features_2'],
			params['seq_model_hyperparams']['num_features_3']
		]
		# Encoder
		params['seq_model_hyperparams']['encoder']['kernel'] = [
			params['seq_model_hyperparams']['encoder']['kernel_1'],
			params['seq_model_hyperparams']['encoder']['kernel_23'],
			params['seq_model_hyperparams']['encoder']['kernel_23']
		]
		params['seq_model_hyperparams']['encoder']['stride'] = [
			params['seq_model_hyperparams']['encoder']['stride_1'],
			params['seq_model_hyperparams']['encoder']['stride_23'],
			params['seq_model_hyperparams']['encoder']['stride_23']
		]
		# Decoder
		params['seq_model_hyperparams']['decoder']['kernel'] = [
			params['seq_model_hyperparams']['decoder']['kernel_1'],
			params['seq_model_hyperparams']['decoder']['kernel_2'],
		]
		params['seq_model_hyperparams']['decoder']['stride'] = [
			params['seq_model_hyperparams']['decoder']['stride_1'],
			params['seq_model_hyperparams']['decoder']['stride_2'],
		]
	return params


def objective(params, checkpoint_dir=None, adata=None):
	"""
	Objective function for Ray Tune
	:param params: Ray Tune will use this automatically
	:param checkpoint_dir: Ray Tune will use this automatically
	:param adata: adata containing train and eval
	"""
	import warnings
	warnings.simplefilter(action='ignore', category=FutureWarning)
	import pandas as pd
	pd.options.mode.chained_assignment = None  # default='warn'

	import sys
	sys.path.append('../../../')
	import tcr_embedding as tcr  # tune needs to reload this module

	random_seed = 42
	import torch
	import numpy as np
	import random
	torch.manual_seed(random_seed)
	np.random.seed(random_seed)
	random.seed(random_seed)

	# Optuna cannot sample within lists, so we have to add those values back into a list
	params = correct_params(params)

	with tune.checkpoint_dir(0) as checkpoint_dir:
		save_path = checkpoint_dir
	# Init Comet-ML
	current_datetime = datetime.now().strftime("%Y%m%d-%H.%M")
	experiment_name = name + '_' + current_datetime
	with open(os.path.dirname(__file__) + '/../comet_ml_key/API_key.txt') as f:
		comet_key = f.read()
	experiment = Experiment(api_key=comet_key, workspace='bcc', project_name=name)
	experiment.log_parameters(params)
	experiment.log_parameters(params['scRNA_model_hyperparams'], prefix='scRNA')
	experiment.log_parameters(params['seq_model_hyperparams'], prefix='seq')
	experiment.log_parameter('experiment_name', experiment_name)
	experiment.log_parameter('save_path', save_path)
	experiment.log_parameter('balanced_sampling', args.balanced_sampling)

	if params['seq_model_arch'] == 'CNN':
		experiment.log_parameters(params['seq_model_hyperparams']['encoder'], prefix='seq_encoder')
		experiment.log_parameters(params['seq_model_hyperparams']['decoder'], prefix='seq_decoder')

	adata = adata[adata.obs['set'] != 'test']  # This needs to be inside the function, ray can't deal with it outside

	if 'single' in args.model and 'separate' not in args.model:
		init_model = tcr.models.single_model.SingleModel
	elif 'moe' in args.model:
		init_model = tcr.models.moe.MoEModel
	elif 'poe' in args.model:
		init_model = tcr.models.poe.PoEModel
	elif 'separate' in args.model:
		init_model = tcr.models.separate_model.SeparateModel
	else:
		init_model = tcr.models.joint_model.JointModel
	# Init Model
	model = init_model(
		adatas=[adata],  # adatas containing gene expression and TCR-seq
		aa_to_id=adata.uns['aa_to_id'],  # dict {aa_char: id}
		seq_model_arch=params['seq_model_arch'],  # seq model architecture
		seq_model_hyperparams=params['seq_model_hyperparams'],  # dict of seq model hyperparameters
		scRNA_model_arch=params['scRNA_model_arch'],
		scRNA_model_hyperparams=params['scRNA_model_hyperparams'],
		zdim=params['zdim'],  # zdim
		hdim=params['hdim'],  # hidden dimension of scRNA and seq encoders
		activation=params['activation'],  # activation function of autoencoder hidden layers
		dropout=params['dropout'],
		batch_norm=params['batch_norm'],
		shared_hidden=params['shared_hidden'],  # hidden layers of shared encoder / decoder
		names=['bcc'],
		gene_layers=[],  # [] or list of str for layer keys of each dataset
		seq_keys=[]  # [] or list of str for seq keys of each dataset
	)

	n_epochs = args.n_epochs * params['batch_size'] // 256  # to have same numbers of iteration
	early_stop = args.early_stop * params['batch_size'] // 256
	epoch2step = 256 / params['batch_size']  # normalization factor of epoch -> step, as one epoch with different batch_size results in different numbers of iterations
	epoch2step *= 1000  # to avoid decimal points, as we multiply with a float number
	save_every = n_epochs // args.num_checkpoints
	# Train Model
	model.train(
		experiment_name=name,
		n_iters=None,
		n_epochs=n_epochs,
		batch_size=params['batch_size'],
		lr=params['lr'],
		losses=params['losses'],  # list of losses for each modality: losses[0] := scRNA, losses[1] := TCR
		loss_weights=params['loss_weights'], # [] or list of floats storing weighting of loss in order [scRNA, TCR, KLD]
		kl_annealing_epochs=None,
		val_split='set',  # float or str, if float: split is determined automatically, if str: used as key for train-val column
		early_stop=early_stop,
		balanced_sampling=args.balanced_sampling,
		validate_every=1,
		save_every=save_every,
		save_path=save_path,
		save_last_model=False,
		num_workers=0,
		device=None,
		comet=experiment,
		tune=tune
	)

	if os.path.exists(os.path.join(save_path, f'{name}_best_rec_model.pt')):
		# Plot UMAPs for model with best reconstruction
		model.load(os.path.join(save_path, f'{name}_best_rec_model.pt'))

		figure_groups = ['patient', 'clonotype', 'cluster', 'cluster_tcr', 'treatment', 'response']
		val_latent = model.get_latent([adata[adata.obs['set'] == 'val']], batch_size=512, metadata=figure_groups)
		figures = tcr.utils.plot_umap_list(val_latent, title=name + '_val_best_recon', color_groups=figure_groups)
		for title, fig in zip(figure_groups, figures):
			experiment.log_figure(figure_name=name + f'_val_{title}', figure=fig, step=model.epoch)
	experiment.end()


parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true', help='Resumes previous training', default=False)
parser.add_argument('--model', type=str, default='single_scRNA')
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--n_epochs', type=int, default=5000)
parser.add_argument('--early_stop', type=int, default=100)
parser.add_argument('--num_samples', type=int, default=100)
parser.add_argument('--num_checkpoints', type=int, default=20)
parser.add_argument('--local_mode', action='store_true', help='Local mode in ray is activated, enables breakpoints')
parser.add_argument('--num_cpu', type=int, default=4)
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--balanced_sampling', type=str, default=None, help='name of the column used to balance')
parser.add_argument('--grid_search', action='store_true')
args = parser.parse_args()

adata = sc.read_h5ad(os.path.dirname(__file__) + '/../data/BCC/06_bcc_highly_var_5000.h5ad')

params = importlib.import_module(f'{args.model}_tune').params
init_params = importlib.import_module(f'{args.model}_tune').init_params

name = f'bcc_tune_{args.model}{args.suffix}'
cwd = os.getcwd()
local_dir = f'{cwd}/../ray_results'
ray.init(local_mode=args.local_mode)

if args.grid_search:
	algo = None
else:
	algo = OptunaSearch(metric='reconstruction', mode='min', points_to_evaluate=init_params)
	algo = ConcurrencyLimiter(algo, max_concurrent=2)

analysis = tune.run(
	tune.with_parameters(objective, adata=adata),
	name=name,
	metric='reconstruction',
	mode='min',
	search_alg=algo,
	num_samples=args.num_samples,
	config=params,
	resources_per_trial={'cpu': args.num_cpu, 'gpu': args.num_gpu},
	local_dir=local_dir,
	trial_dirname_creator=trial_dirname_creator,
	verbose=3,
	resume=args.resume
)