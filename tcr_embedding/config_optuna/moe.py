def suggest_params(trial):
    dropout = trial.suggest_float('dropout', 0, 0.3)  # used twice
    activation = trial.suggest_categorical('activation', ['linear', 'leakyrelu'])  # used for conditional sampling
    mlp_activation = trial.suggest_categorical('mlp_activation', ['leakyrelu', 'relu', 'sigmoid', 'tanh'])  # used for conditional sampling
    rna_hidden = trial.suggest_int('rna_hidden', 200, 2000)  # hdim should be less than rna_hidden
    hdim = trial.suggest_int('hdim', 50, min(rna_hidden, 800))  # shared_hidden should be less than hdim
    shared_hidden = trial.suggest_int('shared_hidden', 30, min(hdim * 2, 500))  # zdim should be less than shared_hidden
    num_layers = trial.suggest_int('num_layers', 1, 3, step=1) if activation == 'leakyrelu' else 1
    rna_num_layers = trial.suggest_int('rna_num_layers', 1, 3, step=1)
    tfmr_encoding_layers = trial.suggest_int('tfmr_encoding_layers', 1, 4, step=1)  # used twice
    loss_weights_kl = trial.suggest_float('loss_weights_kl', 1e-10, 1e-4, log=True)
    loss_weights_seq = trial.suggest_float('loss_weights_tcr', 1e-5, 1, log=True)
    batch_norm = trial.suggest_categorical('batch_norm', [True, False])

    params = {
        'batch_size': trial.suggest_int('batch_size', 128, 1024),
        'learning_rate': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'loss_weights': [1.0, loss_weights_seq, loss_weights_kl],

        'joint': {
            'activation': activation,
            'batch_norm': True,
            'dropout': dropout,
            'hdim': hdim,
            'losses': ['MSE', 'CE'],
            'num_layers': num_layers,
            'shared_hidden': [shared_hidden] * num_layers,
            'zdim': trial.suggest_int('zdim', 5, min(shared_hidden, 50)),
            'c_embedding_dim': 20,
        },
        'rna': {
            'activation': mlp_activation,
            'batch_norm': batch_norm,
            'dropout': dropout,
            'gene_hidden': [rna_hidden] * rna_num_layers,
            'num_layers': rna_num_layers,
            'output_activation': 'linear'
        },
        'tcr': {
            'embedding_size': trial.suggest_categorical('tfmr_embedding_size', [16, 32, 64]),
            'num_heads': trial.suggest_categorical('tfmr_num_heads', [2, 4, 8]),
            'forward_expansion': 4,
            'encoding_layers': tfmr_encoding_layers,
            'decoding_layers': tfmr_encoding_layers,  # encoding_layers is used here too
            'dropout': trial.suggest_float('tfmr_dropout', 0, 0.3, step=0.05),
        },
    }
    return params
