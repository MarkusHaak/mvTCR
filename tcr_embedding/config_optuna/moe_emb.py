def suggest_params(trial):
    dropout = trial.suggest_float('dropout', 0, 0.3)  # used twice
    activation = trial.suggest_categorical('activation', ['linear', 'leakyrelu'])  # used for conditional sampling
    mlp_activation = trial.suggest_categorical('mlp_activation', ['leakyrelu', 'relu', 'sigmoid', 'tanh'])
    rna_hidden = trial.suggest_int('rna_hidden', 200, 2000)  # hdim should be less than rna_hidden
    tcr_hidden = trial.suggest_int('tcr_hidden', 100, 1000)  # hdim should be less than rna_hidden
    hdim = trial.suggest_int('hdim', 50, min(rna_hidden, tcr_hidden, 800), step=2)  # shared_hidden should be less than hdim
    shared_hidden = trial.suggest_int('shared_hidden', 30, min(hdim * 2, 500))  # zdim should be less than shared_hidden
    num_layers = trial.suggest_int('num_layers', 1, 3) if activation == 'leakyrelu' else 1
    rna_num_layers = trial.suggest_int('rna_num_layers', 1, 3)
    tcr_num_layers = trial.suggest_int('tcr_num_layers', 1, 3)
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
            'zdim': trial.suggest_int('zdim', 5, min(shared_hidden, 100)),
            'c_embedding_dim': 20,
        },
        'rna': {
            'activation': mlp_activation,
            'batch_norm': batch_norm,
            'dropout': dropout,
            'hidden': [rna_hidden] * rna_num_layers,
            'num_layers': rna_num_layers,
            'output_activation': 'linear'
        },
        'tcr': {
            'activation': mlp_activation,
            'batch_norm': batch_norm,
            'dropout': dropout,
            'hidden': [tcr_hidden] * tcr_num_layers,
            'num_layers': tcr_num_layers,
            'output_activation': 'linear'
        },
    }
    return params
