# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 18:09:49 2021

@author: Mateus
"""
experiment_params = {
    'version': '0.0.0',
    'game': 'Pong-v0',
    'description': '''
    
    ..
    
    ''',
    
    
    'sample_batch_size': 32,
    'epochs_to_save_results': 10,
    'freq_update_nn': 1000, # *
    'frames_skip': 1,
    'down_sample': 2, #
    'episodes': 1000,
    'freq_save_video': 10,
    
    
    'dirs': {
        'dir_results': 'experiments/Pong/',
        'dir_annotations_experiments': 'experiments/Pong/',
        'dir_videos': 'experiments/Pong/movies/',
        'dir_model': 'experiments/Pong/model/'
    },
    
    'prune_image': {
        'top': 35,
        'bottom': 16,
        'right': 7,
        'left': 7
    },
    
    'params_agent': {
        'memory_size': 150000,
        'min_learning_rate': .00008,
        'max_learning_rate': .001,
        'epochs_interval_lr': 100,
        'gamma': .99,
        'exploration_rate': 1,
        'exploration_min': .07,
        'exploration_decay': 1.0e-5,
        'k_frames': 4,        
        
    },
    
    # Estrutura da Rede Neural Convolucional.
    'structure_neural_network': {
        'input_shape': (80,80),
        'output_activation': 'linear',
        'conv': [
            {
                'filter': 32,
                'kernel_size': (8,8),
                'strides': 4,
                'activation': 'relu',
                'padding': 'valid'
            },
            {
                'filter': 64,
                'kernel_size': (4,4),
                'strides': 2,
                'activation': 'relu',
                'padding': 'valid'
            },
            {
                'filter': 64,
                'kernel_size': (3,3),
                'strides': 1,
                'activation': 'relu',
                'padding': 'valid'
            },
        ],
        
        'neural_network': [
            {
                'neurons': 512,
                'activation': 'relu',
            },
            {
                'neurons': 256,
                'activation': 'relu',
                'kernel_initializer': 'he_uniform'
            },
            {
                'neurons': 64,
                'activation': 'relu',
                'kernel_initializer': 'he_uniform'
            }
        ]
        
    }
}