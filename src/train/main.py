# deep learning libraries
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# other libraries
import os
from tqdm.auto import tqdm

# own modules
from src.train.models import Resnet18, ConvNext, CNNModel
from src.train.utils import accuracy, load_cifar10_data, load_imagenette_data, preprocess_imagenette, set_seed

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


def main() -> None:
    # hyperparameters
    lr = 1e-3
    model_type = 'cnn'
    pretrained = False
    epochs = 50

    # check device
    print(f'device: {device}')
    
    if dataset == 'cifar10':
        train_data, val_data = load_cifar10_data(DATA_PATH[dataset], batch_size=128)
    elif dataset == 'imagenette':
        # preprocess step
        if not os.path.isdir(POSTPROCESS_DATA_PATH[dataset]):
            preprocess_imagenette(DATA_PATH[dataset], POSTPROCESS_DATA_PATH[dataset])

        # load data
        train_data, val_data = load_imagenette_data(POSTPROCESS_DATA_PATH[dataset], batch_size=128)
        
    else:
        raise ValueError('Invdalid dataset value')

    # define model name and tensorboard writer
    name = f'{model_type}_pretrained_{pretrained}'
    writer = SummaryWriter(f'./runs/{dataset}/{name}')

    # define model
    if model_type == 'resnet18':
        model = Resnet18(NUMBER_OF_CLASSES, pretrained).to(device)
    elif model_type == 'convnext':
        model = ConvNext(NUMBER_OF_CLASSES, pretrained).to(device)
    elif model_type == 'cnn':
        model = CNNModel(output_channels=NUMBER_OF_CLASSES).to(device)
    else:
        raise ValueError('Invalid model type')

    # select which layers to activate if pretrained model is loaded
    if pretrained:
        # freeze layers
        for param in model.parameters():
            param.requires_grad_(False)

        # activate classifier and first conv layer if color space is different than rgb
        if model_type == 'resnet18' or model_type == 'convnext':
            model.classifier.requires_grad_(True)
        
        # define optimizer
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr)
        
    else:
        # activate all layers
        for param in model.parameters():
            param.requires_grad_(True)
        
        # define optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # define loss and optimizer
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # define progress bar
    progress_bar = tqdm(range(epochs*(len(train_data) + len(val_data))))

    # epochs loop
    for epoch in range(epochs):
        # train mode
        model.train()

        # initialize vectors
        losses = []
        accuracies = []

        # train step loop
        for images, labels in train_data:
            # pass images and labels to the correct device
            images = images.to(device)
            labels = labels.to(device)

            # compute outputs and loss
            outputs = model(images)
            loss_value = loss(outputs, labels.long())

            # compute gradient and update parameters
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            # add metrics to vectors
            losses.append(loss_value.item())
            accuracies.append(accuracy(outputs, labels))
            
            # progress bar step
            progress_bar.update()

        # write results on tensorboard
        writer.add_scalar('loss/train', np.mean(losses), epoch)
        writer.add_scalar('accuracy/train', np.mean(accuracies), epoch)

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

                # compute outputs and loss
                outputs = model(images)
                loss_value = loss(outputs, labels.long())

                # add metrics to vectors
                losses.append(loss_value.item())
                accuracies.append(accuracy(outputs, labels))
                
                # progress bar step
                progress_bar.update()

            # write results on tensorboard
            writer.add_scalar('accuracy/val', np.mean(accuracies), epoch)

    # save model
    if not os.path.exists(f'./models/{dataset}'):
        os.makedirs(f'./models/{dataset}')
    torch.save(model, f'./models/{dataset}/{name}.pt')
    
    
if __name__ == '__main__':
    main()
