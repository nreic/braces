import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import PIL
from PIL import Image
import torchvision.transforms.v2 as transforms


def t2pil(tensor:torch.Tensor) -> PIL.Image:
    """
    Convert a torch Tensor to a PIL Image.
    :param tensor: torch Tensor to be converted
    :return: PIL Image
    """
    return transforms.ToPILImage()(tensor)


def t2cv(tensor:torch.Tensor, to_gray=False) -> np.array:
    """
    Convert a torch Tensor (image) to cv2 compatible np array in **BGR** and normalized uint8.
    :param tensor: torch Tensor to be converted
    :param to_gray: True if you want to convert to grayscale
    :return: image in np array, BGR, normalized uint8
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input to t2cv is not a torch Tensor.")
    np_arr = np.array(tensor).transpose((1,2,0))
    if to_gray:
        np_arr = cv2.cvtColor(np_arr, cv2.COLOR_RGB2GRAY)
    else:
        np_arr = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)
    np_arr = cv2.normalize(np_arr, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    return np_arr


def to_channel_first_tensor(object):
    if isinstance(object, torch.Tensor):
        return object
    elif isinstance(object, np.ndarray) or isinstance(object, Image.Image):
        return torch.tensor(object).permute(2,0,1)
    else:
        raise ValueError(f"Object could not be converted to Tensor. Only np arrays and PIl images are supported. Object is of type {type(object)}.")


def to_channel_first(array):
    return to_channel_first_tensor(array)


def to_channel_last(tensor):
    """ Converts a tensor to numpy array (cpu) with channel-last representation """
    return tensor.permute(1,2,0).detach().cpu().numpy()


def to_uint_np_channel_last(tensor):
    """ Converts a tensor to numpy array in uint8 with [0,255] values, channel-last """
    return to_channel_last(tensor * 255).astype(np.uint8)


def apply_transform(tensor, transform, mask=None):
    """ Wrapper function to apply albumentation transformations to tensors """
    image = to_channel_last(tensor)
    if mask is not None:
        mask = to_channel_last(mask)
        transformed = transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        return to_channel_first(image), to_channel_first(mask)
    else:
        transformed = transform(image=image)['image']
        return to_channel_first(transformed)


def isPILimage(img):
    return isinstance(img, Image.Image)


def print_img_infos(img, text):
    dtype = shape = max_val = min_val = "Not available"

    # PIL Image
    if isPILimage(img):
        dtype = "PIL Image"
        shape = f"{img.size[::-1]} (Width, Height)"
        if img.mode == "L" or img.mode == "RGB" or img.mode == "RGBA":
            img_np = np.array(img)
            max_val = img_np.max()
            min_val = img_np.min()

    # numpy array
    elif isinstance(img, np.ndarray):
        dtype = img.dtype
        shape = img.shape
        max_val = img.max()
        min_val = img.min()

    # torch tensor
    elif torch.is_tensor(img):
        dtype = img.dtype
        shape = img.shape
        max_val = img.max().item()
        min_val = img.min().item()

    print(
        f"{text}: dtype={dtype}, shape={shape}, max_val={max_val}, min_val={min_val}")


def plot_images_as_grid(images:list, ncols:int =2, figsize=(10,6), save_to_path=None):
    """
    Plots the provided list of images in dictionaries (with other information) as a grid.
    Expects the images as numpy channel_last arrays in RGB format. To convert dimension order, use 'utils.to_channel_last' function.
    :param images: List of dictionaries that contain the image as value of key 'img', title as value of 'title' and optional cmap.
    """
    num_subplots = len(images)
    nrows = divide_and_round_up(num_subplots, ncols)
    figure, axs = plt.subplots(nrows=nrows,
                               ncols=ncols, figsize=figsize)

    for i, subplot in enumerate(images):
        row = i // ncols
        col = i % ncols
        cmap = 'gray' if "cmap" not in subplot else subplot["cmap"]
        image = subplot['img']
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Image passed to 'plot_images_as_grid' is not a numpy array. Please convert before passing as value of key 'img' in the list.")
        if nrows < 2:
            axs[col].imshow(image, cmap=cmap)
            axs[col].set_title(subplot["title"])
            axs[col].axis('off')
        else:
            axs[row, col].imshow(image, cmap=cmap)
            axs[row, col].set_title(subplot["title"])
            axs[row, col].axis('off')

    figure.tight_layout()
    if save_to_path: figure.savefig(save_to_path, dpi=300)
    figure.show()


def divide_and_round_up(divisor, dividend):
    """ Thanks to dlitz from [this SO post](https://stackoverflow.com/a/17511341/23302554) """
    return -(divisor // - dividend)


def use_connected_components(binary_img):
    """ Returns a list of blobs, each with the following information:
    bbox (x_start, y_start, width, height), area, centroid and
    the blobs mask """
    blobs = []
    components = cv2.connectedComponentsWithStats(binary_img)
    (num_labels, masks, stats, centroids) = components

    for label in range(1, num_labels):
        x_start = stats[label, cv2.CC_STAT_LEFT]
        y_start = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        bbox = (x_start, y_start, width, height)
        area = stats[label, cv2.CC_STAT_AREA]
        centroid = centroids[label]
        component_mask = (masks == label).astype(np.uint8)

        blobs.append((bbox, area, centroid, component_mask))

    return blobs
