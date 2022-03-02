# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 18:09:49 2021

@author: Mateus
"""
experiment_params = {
    'version': '0.0.2',
    'game': 'Skiing-v0',
    'description': '''
    
   
    
    ''',
    
    
    'sample_batch_size': 32,
    'epochs_to_save_results': 10,
    'freq_update_nn': 1000, # *
    'frames_skip': 1,
    'down_sample': 2, #
    'episodes': 100000,
    'freq_save_video': 10,
    
    
    'dirs': {
        'dir_results': 'experiments/Skiing-v2/',
        'dir_annotations_experiments': 'experiments/Skiing-v2/',
        'dir_videos': 'experiments/Skiing-v2/movies/',
        'dir_model': 'experiments/Skiing-v2/model/'
    },
    
    'prune_image': {
        'top': 57,  
        'bottom': 1,
        'right': 8,
        'left': 8
    },
    
    'params_agent': {
        'memory_size': 500000,
        'min_learning_rate': .0001,
        'max_learning_rate': .00001,
        'epochs_interval_lr': 300,
        'gamma': .99,
        'exploration_rate': 1,
        'exploration_min': .06,
        'exploration_map': [
                (2500000, (1,.3)), 
                (3500000, (.3,.1)), 
                (5000000, (.1,.06))
            ],
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