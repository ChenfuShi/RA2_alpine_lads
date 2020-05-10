joint_damage_type_train_params = {
    'epochs': 75,
    'batch_size': 64,
    'restart_epochs': 0,
    'lr': 3e-4,
    'wd': 1e-6
}

joint_damage_train_params = {
    'H-J': {
        'epochs': 300,
        'batch_size': 64,
        'steps_per_epoch': 125,
        'split_type': 'balanced',
        'lr': 3e-4,
        'wd': 1e-6
    },
    'H-E': {
        'epochs': 300,
        'batch_size': 64,
        'steps_per_epoch': 125,
        'split_type': 'balanced',
        'lr': 3e-4,
        'wd': 1e-6
    },
    'W-J': {
        'epochs': 300,
        'batch_size': 64,
        'steps_per_epoch': 60,
        'split_type': None,
        'is_wrist': True,
        'lr': 1e-3,
        'wd': 1e-6
    },
    'W-E': {
        'epochs': 300,
        'batch_size': 64,
        'steps_per_epoch': 60,
        'split_type': None,
        'is_wrist': True,
        'lr': 1e-3,
        'wd': 1e-6
    },
    'F-J': {
        'epochs': 300,
        'batch_size': 64,
        'steps_per_epoch': 100,
        'split_type': None,
        'lr': 3e-4,
        'wd': 1e-6
    },
    'F-E': {
        'epochs': 300,
        'batch_size': 64,
        'steps_per_epoch': 120,
        'split_type': None,
        'lr': 3e-4,
        'wd': 1e-6
    },
    'MH-J': {
        'epochs': 300,
        'batch_size': 64,
        'steps_per_epoch': 145,
        'split_type': 'balanced',
        'lr': 3e-4,
        'wd': 1e-6
    },
    'MH-E': {
        'epochs': 300,
        'batch_size': 64,
        'steps_per_epoch': 105,
        'split_type': 'balanced',
        'lr': 3e-4,
        'wd': 1e-6
    },
    'MF-J': {
        'epochs': 300,
        'batch_size': 64,
        'steps_per_epoch': 125,
        'split_type': 'balanced',
        'lr': 3e-4,
        'wd': 1e-6
    },
    'MF-E': {
        'epochs': 300,
        'batch_size': 64,
        'steps_per_epoch': 120,
        'split_type': 'balanced',
        'lr': 1e-3,
        'wd': 1e-6
    }
}
