# deep learning libraries
import torch
import numpy as np

# other libraries
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tqdm.auto import tqdm

# own modules
from src.train.utils import load_cifar10_data, load_imagenette_data, preprocess_imagenette, set_seed
from src.explain.saliency_maps import SaliencyMap, PositiveSaliencyMap, NegativeSaliencyMap, ActiveSaliencyMap, \
    InactiveSaliencyMap

# set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# set all seeds
set_seed(42)

# static variables
DATA_PATH = {'cifar10': 'data/cifar10', 'imagenette': 'data/imagenette'}
NUMBER_OF_CLASSES = 10
METHODS = {'saliency_map': SaliencyMap, 'positive_saliency_map': PositiveSaliencyMap, 
           'negative_saliency_map': NegativeSaliencyMap, 'active_saliency_map': ActiveSaliencyMap, 
           'inactive_saliency_map': InactiveSaliencyMap}
PERCENTAGES = [0, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]


def main() -> None:
    # variables
    dataset = 'imagenette'
    model_type = 'resnet18'
    pretrained = True
    
    # check device
    print(f'device: {device}')
    
    if not os.path.exists(f'{DATA_PATH[dataset]}/saved/{model_type}_{pretrained}'):
    
        if dataset == 'cifar10':
            train_data, val_data = load_cifar10_data(DATA_PATH[dataset], batch_size=128)
        elif dataset == 'imagenette':
            # preprocess step
            if not os.path.isdir(f'{DATA_PATH[dataset]}/postprocess_data'):
                preprocess_imagenette(f'{DATA_PATH[dataset]}/original_data', f'{DATA_PATH[dataset]}/postprocess_data')

            # load data
            train_data, val_data = load_imagenette_data(f'{DATA_PATH[dataset]}/postprocess_data', batch_size=64)
            
        else:
            raise ValueError('Invdalid dataset value')
        
        # load model
        model = torch.load(f'./models/{dataset}/{model_type}_pretrained_{pretrained}.pt')
        model.eval()
        
        # define progress bar
        progress_bar = tqdm(range((len(train_data) + len(val_data))*len(METHODS)*len(PERCENTAGES)))
        
        for loader_name, loader in {'train': train_data, 'val': val_data}.items():
            i = 0
            for images, labels in loader:
                # pass images to device
                images = images.to(device)
                labels = labels.to(device)
                
                # compute outputs
                outputs = model(images)
                
                # check dirs
                if not os.path.exists(f'{DATA_PATH[dataset]}/saved/{model_type}_{pretrained}/{loader_name}/labels'):
                    os.makedirs(f'{DATA_PATH[dataset]}/saved/{model_type}_{pretrained}/{loader_name}/labels')
                if not os.path.exists(f'{DATA_PATH[dataset]}/saved/{model_type}_{pretrained}/{loader_name}/outputs'):
                    os.makedirs(f'{DATA_PATH[dataset]}/saved/{model_type}_{pretrained}/{loader_name}/outputs')
                
                # save original outputs )
                torch.save(labels, f'{DATA_PATH[dataset]}/saved/{model_type}_{pretrained}/'
                        f'{loader_name}/labels/{i}.pt')
                torch.save(outputs, f'{DATA_PATH[dataset]}/saved/{model_type}_{pretrained}/'
                        f'{loader_name}/outputs/{i}.pt')
                
                for method_name, method in METHODS.items():
                    # compute explanations
                    explainer = method(model)
                    saliency_maps = explainer.explain(images)
                    
                    for percentage in PERCENTAGES:
                        saliency_map_sorted, _ = torch.sort(saliency_maps.flatten(start_dim=1), descending=True)
                        
                        # occlude pixels
                        value = saliency_map_sorted[:, round(percentage * saliency_map_sorted.shape[1])]
                        value = value.view(value.shape[0], 1, 1)
                        mask = (saliency_maps > value) * (saliency_maps != 0)
                        mask = mask.unsqueeze(1)
                        mask = mask.repeat(1, 3, 1, 1)
                        
                        for subs_value in [0, 1]:
                            inputs = images.clone()
                            inputs[mask==1] = subs_value
                            
                            # compute outputs
                            outputs = model(inputs)
                            
                            # check dirs
                            if not os.path.exists(f'{DATA_PATH[dataset]}/saved/{model_type}_{pretrained}/'
                                                f'{loader_name}/{method_name}/{subs_value}/{percentage}/outputs'):
                                os.makedirs(f'{DATA_PATH[dataset]}/saved/{model_type}_{pretrained}/{loader_name}/'
                                            f'{method_name}/{subs_value}/{percentage}/outputs')

                            torch.save(outputs, f'{DATA_PATH[dataset]}/saved/{model_type}_{pretrained}/'
                                    f'{loader_name}/{method_name}/{subs_value}/{percentage}/outputs/{i}.pt')
                        
                        # update progress
                        progress_bar.update()
                
                # increment batch index
                i += 1
    
    for subs_value in [0, 1]:
        for loader_name in ['train', 'val']:
            results = {}
            for method_name, method in METHODS.items():
                
                # ignore negative and actives for 0 subs
                if subs_value == 0 and (method_name == 'negative_saliency_map' or method_name == \
                    'inactive_saliency_map'):
                    continue
                
                # ignore positive and actives for 1 subs
                if subs_value == 1 and (method_name == 'positive_saliency_map' or method_name == 'active_saliency_map'):
                    continue
                
                results[method_name] = []
                last_percentage = None
                auc = 0
                for percentage in PERCENTAGES:
                    correct = 0
                    len_loader = 0
                    i = 0
                    for file_name in os.listdir(f'{DATA_PATH[dataset]}/saved/{model_type}_{pretrained}/{loader_name}'
                                                f'/{method_name}/{subs_value}/{percentage}/outputs'):
                        original_outputs = torch.argmax(
                            torch.load(f'{DATA_PATH[dataset]}/saved/{model_type}_{pretrained}/{loader_name}'
                                       f'/outputs/{file_name}'), dim=1)
                        outputs = torch.argmax(
                            torch.load(f'{DATA_PATH[dataset]}/saved/{model_type}_{pretrained}/{loader_name}'
                                       f'/{method_name}/{subs_value}/{percentage}/outputs/{file_name}'), dim=1)
                        
                        correct += (original_outputs == outputs).sum().item()
                        len_loader += outputs.shape[0]
                        
                        i += 1
                    
                    results[method_name].append(correct/len_loader)
                    
                    if last_percentage is None:
                        last_percentage = percentage
                    else:
                        auc += (abs(results[method_name][-1] - results[method_name][-2]) / 2 + \
                            min(results[method_name][-1], results[method_name][-2])) * (percentage - last_percentage)
                        last_percentage = percentage
                
                print(f'{loader_name} AUC {method_name} for {subs_value}: {auc:.2f}')
                    
            # check dir
            if not os.path.exists(f'visualizations/{dataset}/{model_type}_{pretrained}/{loader_name}'):
                os.makedirs(f'visualizations/{dataset}/{model_type}_{pretrained}/{loader_name}')
                    
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plt.plot(PERCENTAGES, np.transpose(np.array(list(results.values()))), marker='o')
            plt.xlabel('pixels deleted [%]')
            plt.ylabel('fidelity')
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
            plt.ylim([0, 1])
            plt.grid()
            plt.legend(list(results.keys()))
            plt.savefig(f'visualizations/{dataset}/{model_type}_{pretrained}/{loader_name}/{subs_value}.pdf', 
                        bbox_inches='tight', pad_inches=0, format='pdf')
            plt.close()
        
    
if __name__ == '__main__':
    main()