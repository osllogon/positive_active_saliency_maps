# deep learning libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from torch.utils.data import DataLoader

# other libraries
import os
from typing import List
from tqdm.auto import tqdm

# own modules
from src.train.utils import accuracy, preprocess_imagenette, set_seed, load_cifar10_data, load_imagenette_data
from src.explain.saliency_maps import SaliencyMap, PositiveSaliencyMap, NegativeSaliencyMap, ActiveSaliencyMap, \
    InactiveSaliencyMap

# set seed
set_seed(42)
    
# set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# static variables
DATA_PATH = {'cifar10': './data/cifar10', 'imagenette': './data/imagenette/original_data'}
POSTPROCESS_DATA_PATH = {'cifar10': None, 'imagenette': './data/imagenette/postprocess_data'}
NUMBER_OF_CLASSES = 10

@torch.no_grad()
def deletion(dataloader: DataLoader, explainer: SaliencyMap, percentages: List[float], 
             descending: bool = True) -> List[float]:

    # initialize results variable
    results = [] 
    
    # iterate through perentages of occlusion
    for percentage in percentages:

        # initialize vector
        accuracies = []

        # val step loop
        for images, labels in dataloader:
            # pass images and labels to the correct device
            images = images.to(device)
            labels = labels.to(device)
            
            # compute saliency maps
            saliency_maps = explainer.explain(images) + explainer.explain(images)
            saliency_map_sorted, _ = torch.sort(saliency_maps.flatten(start_dim=1), descending=descending)
            
            # occlude pixels
            value = saliency_map_sorted[:, round(percentage * saliency_map_sorted.shape[1])]
            value = value.view(value.shape[0], 1, 1)
            mask = (saliency_maps > value) * (saliency_maps != 0) if descending else \
                (saliency_maps < value) * (saliency_maps != 0)
            mask = mask.unsqueeze(1)
            # images[:, 0, :, :][mask] =  0.46064087340445936
            # images[:, 1, :, :][mask] = 0.45542655854304304
            # images[:, 2, :, :][mask] = 0.4273219960606444
            images = images * ~mask

            # compute outputs and loss
            outputs = explainer.model(images)
            
            # add accuracy from the batch
            accuracies.append(images.size(0)*accuracy(outputs, labels))
    
        # compute final accuracy from all dataset
        results.append(np.sum(accuracies)/len(dataloader.dataset))

    return results


if __name__ == '__main__':
    # variables
    dataset = 'imagenette'
    model_type = 'resnet18'
    batch_size = 128
    
    # hyperparameters
    lr = 1e-3
    pretrained = False
    epochs = 50

    if dataset == 'cifar10':
        train_data, val_data = load_cifar10_data(DATA_PATH[dataset], batch_size=batch_size)
    elif dataset == 'imagenette':
        # preprocess step
        if not os.path.isdir(POSTPROCESS_DATA_PATH[dataset]):
            preprocess_imagenette(DATA_PATH[dataset], POSTPROCESS_DATA_PATH[dataset])

        # load data
        train_data, val_data = load_imagenette_data(POSTPROCESS_DATA_PATH[dataset],  batch_size=batch_size)
        
    else:
        raise ValueError('Invdalid dataset value')

    # define model name and tensorboard writer
    model_path = f'./models/{dataset}/{model_type}_pretrained_{pretrained}_lr_{lr}_epochs_{epochs}.pt'

    methods = {'saliency map': SaliencyMap, 'positive saliency map': PositiveSaliencyMap, 
            'negative saliency map': NegativeSaliencyMap, 'active saliency map': ActiveSaliencyMap, 
            'inactive saliency map': InactiveSaliencyMap}

    # initialize percentages
    percentages = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results = {}
    for method_name in list(methods.keys()):
        results[method_name] = deletion(val_data, methods[method_name](torch.load(model_path).to(device)), percentages)
        
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(percentages, np.transpose(np.array(list(results.values()))), marker='o')
    plt.xlabel('pixels deleted [%]')
    plt.ylabel('accuracy')
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    plt.grid()
    plt.legend(list(results.keys()))
    plt.savefig(f'./visualizations/graphs/{model_type}_pretrained_{pretrained}_lr_{lr}_epochs_{epochs}.pdf', 
                bbox_inches='tight', pad_inches=0, format='pdf')