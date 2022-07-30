# deep learning libraries
import torch
import numpy as np

# other libraries
import os
from tqdm.auto import tqdm
from typing import List

# own modules
from src.train.utils import accuracy, preprocess_imagenette, set_seed, load_cifar10_data, load_imagenette_data
from src.explain.saliency_maps import SaliencyMap, PositiveSaliencyMap, NegativeSaliencyMap, ActiveSaliencyMap, \
    InactiveSaliencyMap

# set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# set all seeds
set_seed(42)

# static variables
DATA_PATH = {'cifar10': './data/cifar10', 'imagenette': './data/imagenette/original_data'}
POSTPROCESS_DATA_PATH = {'cifar10': None, 'imagenette': './data/imagenette/postprocess_data'}
NUMBER_OF_CLASSES = 10

# variables
dataset = 'cifar10'


if __name__ == '__main__':
    # hyperparameters
    lr = 1e-3
    model_type = 'resnet18'
    pretrained = False
    epochs = 50

    # check device
    print(f'device: {device}')
    
    if dataset == 'cifar10':
        train_data, val_data = load_cifar10_data(DATA_PATH[dataset], batch_size=64)
    elif dataset == 'imagenette':
        # preprocess step
        if not os.path.isdir(POSTPROCESS_DATA_PATH[dataset]):
            preprocess_imagenette(DATA_PATH[dataset], POSTPROCESS_DATA_PATH[dataset])

        # load data
        train_data, val_data = load_imagenette_data(POSTPROCESS_DATA_PATH[dataset])
        
    else:
        raise ValueError('Invdalid dataset value')

    # define model name and tensorboard writer
    name = f'{model_type}_pretrained_{pretrained}_lr_{lr}_epochs_{epochs}'

    # define model
    model = torch.load(f'./models/{dataset}/{name}.pt').to(device)
            
    results = [] 
            
    for percentage in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

        # evaluation mode
        model.eval()
        with torch.no_grad():

            # initialize vector
            accuracies = []

            # val step loop
            for images, labels in val_data:
                # pass images and labels to the correct device
                images = images.to(device)
                labels = labels.to(device)
                
                method = InactiveSaliencyMap(model)
                saliency_maps = method.explain(images)
                saliency_map_sorted, _ = torch.sort(saliency_maps.flatten(start_dim=1), descending=True)
                
                value = saliency_map_sorted[:, round(percentage * saliency_map_sorted.shape[1])]
                value = value.view(value.shape[0], 1, 1)
                mask = (saliency_maps > value) * (saliency_maps != 0)
                mask = mask.unsqueeze(1)
                images = images * ~mask

                # compute outputs and loss
                outputs = model(images)
                
                accuracies.append(accuracy(outputs, labels))
                
        results.append(np.mean(accuracies))

    print(results)