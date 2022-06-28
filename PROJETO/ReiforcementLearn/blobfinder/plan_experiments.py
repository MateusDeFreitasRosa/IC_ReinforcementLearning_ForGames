# -*- coding: utf-8 -*-
"""
Created on Sun May 22 16:37:30 2022

@author: MateusF
"""

experiment_params = {
    'version': '0.0.1',
    'game': 'Pong-v0',
    'description': '''
        Primeiro experimento com o uso de blobfinder.
    ''',
    
    
    'sample_batch_size': 32,
    'epochs_to_save_results': 10,
    'freq_update_nn': 1000, # *
    'frames_skip': 1,
    'down_sample': 1, #
    'episodes': 100000,
    'freq_save_video': 10,
    
    
    'dirs': {
        'dir_results': 'experiments/Pong-BlobFinder1/',
        'dir_annotations_experiments': 'experiments/Pong-BlobFinder1/',
        'dir_videos': 'experiments/Pong-BlobFinder1/movies/',
        'dir_model': 'experiments/Pong-BlobFinder1/model/'
    },
    
    'prune_image': {
        'top': 34,  
        'bottom': 8,
        'right': 8,
        'left': 15
    },
    
    'params_agent': {
        'memory_size': 200000,
        'min_learning_rate': .00025 ,
        'max_learning_rate': .00008,
        'epochs_interval_lr': 150,
        'gamma': .99,
        'exploration_rate': 1,
        'exploration_min': .07,
        'exploration_map': [
                (200000, (1,.6)), 
                (450000, (.6,.07)), 
                #(5000000, (.1,.06))
            ], # Esta variável diz como a variável de exploração deve ser atualizada. 
        'k_frames': 4,
        'initial_start_size': 5000
    },
    
    # Estrutura da Rede Neural Convolucional.
    'structure_neural_network': {
        'input_shape': (80,80),
        'output_activation': 'linear',
#        'conv': [
#            {
#                'filter': 32,
#                'kernel_size': (8,8),
#                'strides': 4,
#                'activation': 'relu',
#                'padding': 'valid'
#            },
#            {
#                'filter': 64,
#                'kernel_size': (4,4),
#                'strides': 2,
#                'activation': 'relu',
#                'padding': 'valid'
#            },
#            {
#                'filter': 64,
#                'kernel_size': (3,3),
#                'strides': 1,
#                'activation': 'relu',
#                'padding': 'valid'
#            },
#        ],
        
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