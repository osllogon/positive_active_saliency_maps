# deep learning libraries
import torch
import torchvision


class Resnet18(torch.nn.Module):
    """
    This class is a model based in resnet18 for classification
    
    Attributes
    ----------
    cnn_net : torch.nn.Module
        convolutional layers part of the model
    classifier : torch.nn.Linear
        final linear layer for classification
        
    Methods
    -------
    forward -> torch.Tensor
    """

    def __init__(self, output_channels: int = 10, pretrained: bool = True):
        """
        Constructor of Resnet18 class
        
        Parameters
        ----------
        input_channels : int, optional
            input channels for the first conv layer. Deafult value: 3
        pretrained : bool, Optional
            bool that indicates if the resnet18 is pretrained with Imagenet data
        """

        # call super class constructor
        super().__init__()

        # load pretrained resnet18
        if pretrained:
            self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        else:
            self.model = torchvision.models.resnet18(weights=None)

        # define classifier layer
        self.model.fc = torch.nn.Linear(512, output_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the outputs of the neural net
        
        Parameters
        ----------
        inputs : torch.Tensor
            batch of images. Dimensions: [batch, channels, height, width]
            
        Returns
        -------
        torch.Tensor
            batch of logits. Dimensions: [batch, number of classes]
        """

        # compute output
        outputs = self.model(inputs)

        return outputs
    
    
class EfficientNetV2(torch.nn.Module):
    """
    This class is a model based in convnext for classification
    
    Attributes
    ----------
    cnn_net : torch.nn.Module
        convolutional layers part of the model
    classifier : torch.nn.Linear
        final linear layer for classification
        
    Methods
    -------
    forward -> torch.Tensor
    """

    def __init__(self, output_channels: int = 10, pretrained: bool = True):
        """
        Constructor of ConvNextBase class
        
        Parameters
        ----------
        input_channels : int, optional
            input channels for the first conv layer. Deafult value: 3
        pretrained : bool, Optional
            bool that indicates if the resnet18 is pretrained with Imagenet data
        """

        # call super class constructor
        super().__init__()

        # load pretrained convnext_base
        if pretrained:
            self.model = torchvision.models.efficientnet_v2_s(weights=
                                                              torchvision.models.EfficientNet_V2_S_Weights.DEFAULT)
        else:
            self.model = torchvision.models.efficientnet_v2_s(weights=None)

        # define classifier layer
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3), 
            torch.nn.Linear(1280, output_channels)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the outputs of the neural net
        
        Parameters
        ----------
        inputs : torch.Tensor
            batch of images. Dimensions: [batch, channels, height, width]
            
        Returns
        -------
        torch.Tensor
            batch of logits. Dimensions: [batch, number of classes]
        """

        # compute output
        outputs = self.model(inputs)

        return outputs

