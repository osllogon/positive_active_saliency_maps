# deep learning libraries
import torch
import torch.nn.functional as F

    
class SaliencyMap:
    
    def __init__(self, model: torch.nn.Module) -> None:
        """
        Constructor of SaliencyMap class

        Parameters
        ----------
        model : torch.nn.Module
            model for classifying images
        """
        
        self.model = model
        
    @torch.enable_grad()
    def _compute_gradients(self, images: torch.Tensor) -> torch.Tensor:
        """
        This method computes gradients of images
        
        Parameters
        ----------
        images : torch.Tensor
            batch of images. Dimensions [batch, channels, height, width]
        Returns
        -------
        torch.Tensor
             batch of gradients. Dimensions: [batch, channels, height, width]
        """

        # forward pass
        inputs = images.clone()
        inputs.requires_grad_(True)
        outputs = self.model(inputs)
        max_scores, _ = torch.max(outputs, dim=1)

        # clear previous gradients and backward pass
        self.model.zero_grad()
        max_scores.backward(torch.ones_like(max_scores))

        return inputs.grad
    
    # overriding abstract method
    @torch.no_grad()
    def explain(self, images: torch.Tensor) -> torch.Tensor:
        """
        This method computes saliency maps
        
        Parameters
        ----------
        images : torch.Tensor
            batch of images. Dimensions: [batch, channels, height, width]
            
        Returns
        -------
        torch.Tensor
            batch of saliency maps. Dimensions: [batch, height, width]
        """

        # compute saliency maps
        gradients = self._compute_gradients(images)
        saliency_maps, _ = torch.max(torch.abs(gradients), dim=1)

        # normalize between 0 and 1
        min_ = torch.amin(saliency_maps, dim=(1, 2), keepdim=True)
        max_ = torch.amax(saliency_maps, dim=(1, 2), keepdim=True)
        saliency_maps = (saliency_maps - min_) / (max_ - min_)

        return saliency_maps
    
class SmoothGradSaliencyMap(SaliencyMap):
    
    def __init__(self, model: torch.nn.Module) -> None:
        """
        Constructor of SmoothGradSaliencyMap class

        Parameters
        ----------
        model : torch.nn.Module
            model for classifying images
        """
        
        super().__init__(model)
    
    # overriding method
    def explain(self, images: torch.Tensor) -> torch.Tensor:
        """
        This method computes SmoothGrad Saliency Maps
        
        Parameters
        ----------
        images : torch.Tensor
            batch of images. Dimensions: [bath size, channels, height, width]
            
        Returns
        -------
            batch of saliency maps. Dimensions: [batch size, height, width]
        """
        
        # get device from images
        device = images.device

        # compute inputs with noise
        min_ = torch.amin(images, dim=(1, 2, 3), keepdim=True)
        max_ = torch.amax(images, dim=(1, 2, 3), keepdim=True)
        std = (max_ - min_) * self.noise_level * torch.ones(self.sample_size, *images.size()).to(device)
        noise = torch.normal(mean=0, std=std)
        inputs = images.clone().unsqueeze(0)
        inputs = inputs + noise

        # create gradients tensor
        gradients = torch.zeros_like(inputs)

        # compute gradients for each noise batch
        for i in range(inputs.size(0)):
            # clone batch
            inputs_batch = inputs[i].clone()

            # pass the noise batch through the model
            with torch.enable_grad():
                inputs_batch.requires_grad_()
                outputs = self.model(inputs_batch)
                max_scores, _ = torch.max(outputs, dim=1)

                # compute gradients
                self.model.zero_grad()
                max_scores.backward(torch.ones_like(max_scores))
                gradients[i] = inputs_batch.grad

        # create smoothgrad saliency maps
        saliency_maps, _ = torch.max(torch.abs(gradients), dim=2)
        saliency_maps = torch.sum(saliency_maps, dim=0) / self.sample_size

        # normalize between 0 and 1
        min_ = torch.amin(saliency_maps, dim=(1, 2), keepdim=True)
        max_ = torch.amax(saliency_maps, dim=(1, 2), keepdim=True)
        saliency_maps = (saliency_maps - min_) / (max_ - min_)

        return saliency_maps
    
    
class PositiveSaliencyMap(SaliencyMap):
    """
    This class creates positive saliency maps visualizations. This class inherits from SaliencyMap class
    
    Attributes
    ----------
    model : torch.Tensor
        neural network used for classify images
        
    Methods
    -------
    explain -> torch.Tensor
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Constructor of PositiveSaliencyMap
        
        Parameters
        ----------
        model : torch.nn.Module
            model for classifying images
        """

        # call super class constructor
        super().__init__(model)

    # overriding method
    @torch.no_grad()
    def explain(self, images: torch.Tensor) -> torch.Tensor:
        """
        This method computes Positive Saliency Maps
        
        Parameters
        ----------
        images : torch.Tensor
            batch of images. Dimensions: [batch, channels, height, width]
            
        Returns
        -------
        torch.Tensor
            batch of positive saliency maps. Dimensions: [batch, height, width]
        """

        # compute positive saliency maps
        gradients = self._compute_gradients(images)
        positive_gradients = F.relu(gradients)
        saliency_maps, _ = torch.max(positive_gradients, dim=1)

        # normalize between 0 and 1
        min_ = torch.amin(saliency_maps, dim=(1, 2), keepdim=True)
        max_ = torch.amax(saliency_maps, dim=(1, 2), keepdim=True)
        saliency_maps = (saliency_maps - min_) / (max_ - min_)

        return saliency_maps


class NegativeSaliencyMap(SaliencyMap):
    """
    This class creates negative saliency maps visualizations. This class inherits from SaliencyMap class
    
    Attributes
    ----------
    model : torch.Tensor
        neural network used for classify images
        
    Methods
    -------
    explain -> torch.Tensor
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Constructor of NegativeSaliencyMap
        
        Parameters
        ----------
        model : torch.nn.Module
            model for classifying images
        """

        # call super class constructor
        super().__init__(model)

    # overriding method
    @torch.no_grad()
    def explain(self, images: torch.Tensor) -> torch.Tensor:
        """
        This method computes Negative Saliency Maps
        
        Parameters
        ----------
        images : torch.Tensor
            batch of images. Dimensions: [batch, channels, height, width]
            
        Returns
        -------
        torch.Tensor
            batch of positive saliency maps. Dimensions: [batch, height, width]
        """

        # compute negative saliency maps
        gradients = self._compute_gradients(images)
        negative_gradients = F.relu(-gradients)
        saliency_maps, _ = torch.max(negative_gradients, dim=1)

        # normalize between 0 and 1
        min_ = torch.amin(saliency_maps, dim=(1, 2), keepdim=True)
        max_ = torch.amax(saliency_maps, dim=(1, 2), keepdim=True)
        saliency_maps = (saliency_maps - min_) / (max_ - min_)

        return saliency_maps


class ActiveSaliencyMap(SaliencyMap):
    """
    This class creates active saliency maps visualizations. This class inherits from SaliencyMap class
    
    Attributes
    ----------
    model : torch.Tensor
        neural network used for classify images
        
    Methods
    -------
    explain -> torch.Tensor
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Constructor of ActiveSaliencyMap
        
        Parameters
        ----------
        model : torch.nn.Module
            model for classifying images
        """

        # call super class constructor
        super().__init__(model)

    # overriding method
    @torch.no_grad()
    def explain(self, images: torch.Tensor) -> torch.Tensor:
        """
        This method computes Active Saliency Maps
        
        Parameters
        ----------
        images : torch.Tensor
            batch of images. Dimensions: [batch, channels, height, width]
            
        Returns
        -------
        torch.Tensor
            batch of active saliency maps. Dimensions: [batch, height, width]
        """

        # compute original saliency maps
        saliency_maps = self._compute_gradients(images)
        number_of_classes = self.model(images).size(0)
        class_indexes = torch.argmax(self.model(images), dim=1).reshape(number_of_classes, 1, 1, 1)

        # iterate over different classes
        for i in range(number_of_classes):
            # clone images
            inputs = images.clone()

            with torch.enable_grad():
                # forward pass
                inputs.requires_grad_()
                outputs = self.model(inputs)
                max_scores = outputs[:, i]

                # clear previous gradients and backward pass
                self.model.zero_grad()
                torch.sum(max_scores).backward()

            # update saliency maps
            gradients = inputs.grad
            mask = (class_indexes != i) * (saliency_maps <= gradients)
            saliency_maps[mask] = 0

        # compute absolute value of saliency maps
        saliency_maps, _ = torch.max(torch.abs(saliency_maps), dim=1)

        # normalize between 0 and 1
        min_ = torch.amin(saliency_maps, dim=(1, 2), keepdim=True)
        max_ = torch.amax(saliency_maps, dim=(1, 2), keepdim=True)
        saliency_maps = (saliency_maps - min_) / (max_ - min_)

        return saliency_maps


class InactiveSaliencyMap(SaliencyMap):
    """
    This class creates inactive saliency maps visualizations. This class inherits from SaliencyMap class
    
    Attributes
    ----------
    model : torch.Tensor
        neural network used for classify images
        
    Methods
    -------
    explain -> torch.Tensor
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Constructor of InactiveSaliencyMap
        
        Parameters
        ----------
        model : torch.nn.Module
            model for classifying images
        """

        # call super class constructor
        super().__init__(model)

    # overriding method
    @torch.no_grad()
    def explain(self, images: torch.Tensor) -> torch.Tensor:
        """
        This method computes Inactive Saliency Maps
        
        Parameters
        ----------
        images : torch.Tensor
            batch of images. Dimensions: [batch, channels, height, width]
            
        Returns
        -------
        torch.Tensor
            batch of inactive saliency maps. Dimensions: [batch, height, width]
        """

        # compute original saliency maps
        saliency_maps = self._compute_gradients(images)
        number_of_classes = self.model(images).size(1)
        class_indexes = torch.argmax(self.model(images), dim=1).reshape(images.size(0), 1, 1, 1)

        # iterate over different classes
        for i in range(number_of_classes):
            # clone images
            inputs = images.clone()

            with torch.enable_grad():
                # forward pass
                inputs.requires_grad_()
                outputs = self.model(inputs)
                max_scores = outputs[:, i]

                # clear previous gradients and backward pass
                self.model.zero_grad()
                torch.sum(max_scores).backward()

            # update saliency maps
            gradients = inputs.grad
            mask = (class_indexes != i) * (saliency_maps >= gradients)
            saliency_maps[mask] = 0

        # compute absolute value of saliency maps
        saliency_maps, _ = torch.max(torch.abs(saliency_maps), dim=1)

        # normalize betwwen 0 and 1
        min_ = torch.amin(saliency_maps, dim=(1, 2), keepdim=True)
        max_ = torch.amax(saliency_maps, dim=(1, 2), keepdim=True)
        saliency_maps = (saliency_maps - min_) / (max_ - min_)

        return saliency_maps
    
