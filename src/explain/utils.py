# deep learning libraries
import torch
import numpy as np

# other libraries
from typing import List


def auc(vector: List[float], len_between: float = 1.0) -> float:
    """
    This function measures the area under the curve.

    Args:
        vector: list of values.
        len_between: legth between elements of the vector.
            Defaults to 1.0.

    Returns:
        the area under the curve.
    """

    area = 0.0
    for i in range(1, len(vector)):
        if vector[i] > vector[i - 1]:
            area += (
                len_between * vector[i - 1]
                + len_between * (vector[i] - vector[i - 1]) / 2
            )
        else:
            area += (
                len_between * vector[i] + len_between * (vector[i - 1] - vector[i]) / 2
            )

    return area


def format_image(image: torch.Tensor) -> np.ndarray:
    """
    This function formats a torch image to be able to visualize it
    with matplotlib imshow method.

    Args:
        image: image in torch format.
            Dimensions: [channels, height, width].

    Returns:
        image in numpy format. Dimensions: [height, width, channels]
    """

    image = torch.swapaxes(image, 0, 2)
    image_numpy: np.ndarray = torch.swapaxes(image, 0, 1).detach().cpu().numpy()

    return image_numpy
