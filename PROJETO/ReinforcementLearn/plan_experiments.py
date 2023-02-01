# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 18:09:49 2021

@author: Mateus
"""
experiment_params = {
    'version': '0.0.1',
    'game': 'Pong-v0',
    'description': '''
        Experimento base utilizando a mesma topologia de rede neural usada por Minih, porém utilizando
        diferentes e melhores parametros que encontramos.
    ''',
    
    
    'sample_batch_size': 32,
    'epochs_to_save_results': 10,
    'freq_update_nn': 1000, # *
    'frames_skip': 1,
    'down_sample': 1, #
    'episodes': 100000,
    'freq_save_video': 10,
    
    
    'dirs': {
        'dir_results': 'experiments/Pong-useToPaper01/',
        'dir_annotations_experiments': 'experiments/Pong-useToPaper01/',
        'dir_videos': 'experiments/Pong-useToPaper01/movies/',
        'dir_model': 'experiments/Pong-useToPaper01/model/'
    },
    
    'prune_image': {
        'top': 34,  
        'bottom': 16,
        'right': 7,
        'left': 7
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