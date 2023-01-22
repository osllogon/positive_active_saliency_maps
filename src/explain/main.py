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
from src.explain.utils import format_image

# set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# set all seeds and number of threads
set_seed(42)
torch.set_num_threads(8)

# static variables
DATA_PATH = {'cifar10': 'data/cifar10', 'imagenette': 'data/imagenette'}
NUMBER_OF_CLASSES = 10
METHODS = {'saliency_map': SaliencyMap, 'positive_saliency_map': PositiveSaliencyMap, 
           'negative_saliency_map': NegativeSaliencyMap, 'active_saliency_map': ActiveSaliencyMap, 
           'inactive_saliency_map': InactiveSaliencyMap}
PERCENTAGES = [0, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def main() -> None:
    # variables
    generate_examples = False
    generate_graphs = True
    
    # hyperparameters
    dataset = 'cifar10'
    model_type = 'resnet18'
    pretrained = False
    
    # load model
    model = torch.load(f'./models/{dataset}/{model_type}_pretrained_{pretrained}.pt').to(device)
    model.eval()
    
    # check device
    print(f'device: {device}')
    
    if generate_examples:
        # define paths 
        examples_path = f'{DATA_PATH[dataset]}/examples'
        visualizations_path = 'visualizations/images'
        
        # load data
        if dataset == 'cifar10':
            train_data, val_data = load_cifar10_data(DATA_PATH[dataset], batch_size=1)
        elif dataset == 'imagenette':
            # preprocess step
            if not os.path.isdir(f'{DATA_PATH[dataset]}/postprocess_data'):
                preprocess_imagenette(f'{DATA_PATH[dataset]}/original_data', f'{DATA_PATH[dataset]}/postprocess_data')

            # load data
            train_data, val_data = load_imagenette_data(f'{DATA_PATH[dataset]}/postprocess_data', batch_size=1)
            
        else:
            raise ValueError('Invalid dataset value')
        
        iterator = iter(val_data)
        image, label = next(iterator)
        height = image.shape[2]
        width = image.shape[3]
        
        # create directory for saving correct examples if it does not exist
        if not os.path.isdir(examples_path):
            os.makedirs(examples_path)

        # if the examples does not exist yet create them
        if len(os.listdir(examples_path)) == 0:
            # initialize correct and wrong examples vectors
            examples = NUMBER_OF_CLASSES * [None]

            # iter over the dataset looking for correct examples
            for image, label in val_data:
                image = image.to(device)
                label = label.to(device)
                output = torch.argmax(model(image), dim=1)

                # ser correct examples values
                if output == label:
                    if examples[label] is None:
                        examples[label] = image

            # write examples in memory
            i = 0
            for example in examples:
                torch.save(example, f'{examples_path}/{i}.pt')
                i += 1

        # create tensors for examples
        examples = torch.zeros((NUMBER_OF_CLASSES, 3, height, width)).to(device)

        # load examples
        for i in range(NUMBER_OF_CLASSES):
            examples[i] = torch.load(f'{examples_path}/{i}.pt').squeeze(0).to(device)
            
        # check if visualization path is created
        if not os.path.isdir(f'{visualizations_path}/examples/{dataset}/{model_type}_{pretrained}'):
            os.makedirs(f'{visualizations_path}/examples/{dataset}/{model_type}_{pretrained}')
        
        # create and save examples images
        figures = []
        for i in range(examples.shape[0]):
            figure = plt.figure()
            plt.axis('off')
            plt.imshow(format_image(examples[i]), cmap='hot')
            plt.savefig(f'{visualizations_path}/examples/{dataset}/{model_type}_{pretrained}/{i}.png', 
                        bbox_inches='tight', pad_inches=0, format='png', dpi=300)
            plt.close()
            figures.append(figure)
            
        # iterate over methods
        for method_name, method in METHODS.items():
            # compute explanations
            explainer = method(model)
            saliency_maps = explainer.explain(examples)
            
            # check if visualization path is created
            if not os.path.isdir(f'{visualizations_path}/saliency_maps/{method_name}/'
                                    f'{dataset}/{model_type}_{pretrained}'):
                os.makedirs(f'{visualizations_path}/saliency_maps/{method_name}/'
                            f'{dataset}/{model_type}_{pretrained}')
            
            # create and save examples images
            figures = []
            for i in range(examples.shape[0]):
                figure = plt.figure()
                plt.axis('off')
                plt.imshow(saliency_maps[i].detach().cpu().numpy(), cmap='hot')
                plt.savefig(f'{visualizations_path}/saliency_maps/{method_name}/{dataset}/'
                            f'{model_type}_{pretrained}/{i}.png', bbox_inches='tight', pad_inches=0, 
                            format='png', dpi=300)
                plt.close()
                figures.append(figure)
            
            for percentage in PERCENTAGES:
                saliency_map_sorted, _ = torch.sort(saliency_maps.flatten(start_dim=1), descending=True)
                
                # occlude pixels
                value = saliency_map_sorted[:, round(percentage * saliency_map_sorted.shape[1])]
                value = value.view(value.shape[0], 1, 1)
                mask = (saliency_maps > value) * (saliency_maps != 0)
                mask = mask.unsqueeze(1)
                mask = mask.repeat(1, 3, 1, 1)
                
                inputs = examples.clone()
                inputs[mask==1] = 0
                    
                # check if visualization path is created
                if not os.path.isdir(f'{visualizations_path}/examples_filtered/{method_name}/{percentage}/'
                                        f'{dataset}/{model_type}_{pretrained}'):
                    os.makedirs(f'{visualizations_path}/examples_filtered/{method_name}/{percentage}/'
                                f'{dataset}/{model_type}_{pretrained}')
                    
                # create and save examples images
                figures = []
                for i in range(examples.shape[0]):
                    figure = plt.figure()
                    plt.axis('off')
                    plt.imshow(format_image(inputs[i]), cmap='hot')
                    plt.savefig(f'{visualizations_path}/examples_filtered/{method_name}/{percentage}/{dataset}/'
                                f'{model_type}_{pretrained}/{i}.png', bbox_inches='tight', pad_inches=0, 
                                format='png', dpi=300)
                    plt.close()
                    figures.append(figure)
            
        
    if generate_graphs:
        # generate data if it does not exist to avoid doing the computation again
        if not os.path.exists(f'{DATA_PATH[dataset]}/saved/{model_type}_{pretrained}'):
            # load data
            if dataset == 'cifar10':
                train_data, val_data = load_cifar10_data(DATA_PATH[dataset], batch_size=128, num_workers=4)
            elif dataset == 'imagenette':
                # preprocess step
                if not os.path.isdir(f'{DATA_PATH[dataset]}/postprocess_data'):
                    preprocess_imagenette(f'{DATA_PATH[dataset]}/original_data', 
                                          f'{DATA_PATH[dataset]}/postprocess_data')

                # load data
                train_data, val_data = load_imagenette_data(f'{DATA_PATH[dataset]}/postprocess_data', batch_size=128, 
                                                            num_workers=4)
                
            else:
                raise ValueError('Invalid dataset value')
            
            # define progress bar
            progress_bar = tqdm(range((len(train_data) + len(val_data))*len(METHODS)*len(PERCENTAGES)))
            
            # iterate over datasets
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
                    if not os.path.exists(f'{DATA_PATH[dataset]}/saved/{model_type}_{pretrained}/'
                                          f'{loader_name}/outputs'):
                        os.makedirs(f'{DATA_PATH[dataset]}/saved/{model_type}_{pretrained}/{loader_name}/outputs')
                    
                    # save original outputs )
                    torch.save(labels, f'{DATA_PATH[dataset]}/saved/{model_type}_{pretrained}/'
                            f'{loader_name}/labels/{i}.pt')
                    torch.save(outputs, f'{DATA_PATH[dataset]}/saved/{model_type}_{pretrained}/'
                            f'{loader_name}/outputs/{i}.pt')
                    
                    # iterate over methods
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
        
        # create graphs
        for subs_value in [0, 1]:
            for loader_name in ['train', 'val']:
                results = {}
                for method_name, method in METHODS.items():
                    # ignore negative and actives for 0 subs
                    if subs_value == 0 and (method_name == 'negative_saliency_map' or method_name == \
                        'inactive_saliency_map'):
                        continue
                    
                    # ignore positive and actives for 1 subs
                    if subs_value == 1 and \
                        (method_name == 'positive_saliency_map' or method_name == 'active_saliency_map'):
                        continue
                    
                    results[method_name] = []
                    last_percentage = None
                    auc = 0
                    for percentage in PERCENTAGES:
                        correct = 0
                        len_loader = 0
                        i = 0
                        for file_name in os.listdir(f'{DATA_PATH[dataset]}/saved/'
                                                    f'{model_type}_{pretrained}/{loader_name}'
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
                                min(results[method_name][-1], results[method_name][-2])) * \
                                    (percentage - last_percentage)
                            last_percentage = percentage
                    
                    print(f'{loader_name} AUC {method_name} for {subs_value}: {auc:.2f}')
                        
                # check dir
                if not os.path.exists(f'visualizations/graphs/{dataset}/{model_type}_{pretrained}/{loader_name}'):
                    os.makedirs(f'visualizations/graphs/{dataset}/{model_type}_{pretrained}/{loader_name}')
                        
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                plt.plot(PERCENTAGES, np.transpose(np.array(list(results.values()))), marker='o')
                plt.xlabel('pixels deleted [%]')
                plt.ylabel('fidelity')
                ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
                plt.ylim([0, 1])
                plt.grid()
                plt.legend(list(results.keys()))
                plt.savefig(f'visualizations/graphs/{dataset}/{model_type}_{pretrained}/{loader_name}/{subs_value}.pdf', 
                            bbox_inches='tight', pad_inches=0, format='pdf')
                plt.close()
        
    
if __name__ == '__main__':
    main()