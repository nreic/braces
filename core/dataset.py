import numpy as np
import random
import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.v2 as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from imutils import paths as pths
import matplotlib.pyplot as plt
from PIL import Image

from core.paths import train_dir, mini_dataset
from core.utils import to_channel_last, apply_transform, plot_images_as_grid


class TeethDataset (Dataset):
    label = 8
    means = [x / 255 for x in [134.4062063598633, 120.53171859741211, 113.5545482635498]]
    stds = [x / 255 for x in [67.48241500264352, 63.97537158546411, 63.60831609335822]]

    def __init__(self, base_dir, transformations=None, normalize_images=True, with_masks=True):
        """
        :param base_dir: path to the directory that contains the images and annotations (only if with_masks=True) **folders**
        :param transformations: composed torch transformations
        :param normalize_images: (default True) if the images should be normalized, i.e. during training. If dataset is created only for visualization purposes, make sure to set False.
        :param with_masks: (default True) if the images have corresponding masks, necessary for training. If dataset is created for inference, set with_masks False.
        """
        self.image_dir = os.path.join(base_dir, "images")
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(
                f"directory '{base_dir}' does not exist or does not contain 'images' folder.")
        self.image_paths = sorted(list(pths.list_images(self.image_dir)))

        if with_masks:
            self.mask_dir = os.path.join(base_dir, "annotations")
            if not os.path.exists(self.mask_dir):
                raise FileNotFoundError(
                    f"directory '{base_dir}' does not exist or does not contain 'annotations' folder. If you wanted to create a dataset without annotations, set with_masks to False.")
            self.mask_paths = sorted(list(pths.list_images(self.mask_dir)))
        self.is_with_masks = with_masks

        self.transformations = transformations
        self.normalize = A.Normalize(mean=TeethDataset.means, std=TeethDataset.stds) if normalize_images else None

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, item: int):
        """ Returns image (and corresponding mask if with_masks=True) from dataset in channel-first """
        image = cv2.imread(self.image_paths[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.is_with_masks:
            mask = cv2.imread(self.mask_paths[item], cv2.IMREAD_GRAYSCALE)
            mask[mask < TeethDataset.label] = 0
            mask[mask == TeethDataset.label] = 1

        if self.transformations:
            if self.is_with_masks:
                transformed = self.transformations(image=image, mask=mask)
                image, mask = transformed['image'], transformed['mask'] # channel-last
            else:
                image = self.transformations(image=image)['image']

        if self.normalize:
            image = self.normalize(image=image)['image']

        image = ToTensorV2()(image=image)['image']
        image = transforms.ToDtype(torch.float32, scale=True)(image)  # channel-first

        if self.is_with_masks:
            mask = ToTensorV2()(image=mask)['image']
            mask = transforms.ToDtype(torch.float32)(mask)
            return image, mask

        return image

    def get_random(self):
        """ Returns randomly picked image and corresponding mask from this dataset """
        index = random.choice(range(self.__len__()))
        return *self.__getitem__(index), index

    def print_infos(self):
        print(f"[INFO] Number of images in the dataset: {self.__len__()}")
        sample, _ = self.get_random()
        if self.is_with_masks:
            print(f"Dataset contains images and masks.")
            print(f"Random image: {sample[0].shape}, mask: {sample[1].shape}")
        else:
            print(f"Dataset only contains images, no masks.")
            print(f"Random image: {sample.shape}")


def compute_statistics_per_channel(image_dir, num_images, func, target_size=(256, 256)):
    image_files = os.listdir(image_dir)[:num_images]
    images = []
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path)
        image = image.resize(target_size)
        image_arr = np.array(image)
        images.append(image_arr)
    images = np.array(images)
    statistics = []
    for channel in range(images.shape[-1]):
        statistics.append(func(images[:,:,:,channel].flatten()))

    return statistics


def overlay_mask_on_image(image, mask, color=(0, 255, 0), denorm=True):
    """ creates masked image from image and mask, considered in channel-first """
    if denorm: image = denormalize(image)
    if not image.dtype == torch.uint8:
        image = transforms.ToDtype(torch.uint8, scale=True)(image)
    if not mask.dtype == torch.bool:
        mask = transforms.ToDtype(torch.bool)(mask)
    masked_image = draw_segmentation_masks(image, mask, alpha=0.3, colors=color)
    return masked_image


def denormalize(normalized_image):
    denorm_image = normalized_image.clone()
    channel_means = TeethDataset.means
    channel_stds = TeethDataset.stds
    for channel, (mean, std) in enumerate(zip(channel_means, channel_stds)):
        denorm_image[channel] = denorm_image[channel] * std + mean
    return denorm_image


def get_random_samples(num_images:int, base_dir:str, size:int=None):
    """
    Returns a random sample pair (or several) from the directory, optional resize.
    :param num_images: The number of sample pairs that you want to get
    :param base_dir: The base directory that contains 'images' and 'annotations' folder with data
    :param size: to crop the samples (smaller edge) to, if no resize is wanted set size=None
    :return: Tuple of tensors: image and corresponding mask (channel-first)
    """
    transformations = A.Resize(height=size, width=size) if size is not None else None
    ds = TeethDataset(base_dir, transformations, normalize_images=False)
    samples = []
    for i in range(num_images):
        image, mask, _ = ds.get_random()
        samples.append((image, mask))
    return samples


def visualize_transformations(transformations:list, directory, ncols=2, masked=False):
    [(image, mask)] = get_random_samples(1, directory)
    transformed_image_list = [{"img": to_channel_last(image),
                               "cmap": None,
                               "title": "Original Image"}]
    all_transforms = image.clone()
    all_transforms_mask = mask.clone()
    for transformation in transformations:
        trans_img, trans_mask = apply_transform(image, transformation, mask)
        all_transforms, all_transforms_mask = apply_transform(all_transforms, transformation, all_transforms_mask)
        if masked:
            trans_img = overlay_mask_on_image(trans_img, trans_mask, denorm=False)
        transformed_image_list.append({"img": to_channel_last(trans_img),
                                       "cmap": None,
                                       "title": str(transformation).split('(')[0]})
    if masked:
        all_transforms = overlay_mask_on_image(all_transforms, all_transforms_mask,
                                               denorm=False)
    transformed_image_list.append({"img": to_channel_last(all_transforms),
                                   "cmap": None,
                                   "title": "All Transformations applied"})
    plot_images_as_grid(transformed_image_list, ncols=ncols)


def test_transformations():
    transformations = [A.HorizontalFlip(p=.5),
                       A.Rotate(45, p=.4),
                       A.Perspective(p=.4),
                       A.RandomResizedCrop(height=128,
                                           width=128,
                                           scale=(.4, 1.),
                                           ratio=(1, 1),
                                           p=1.),
                       A.RandomBrightnessContrast(p=.5),
                       A.ColorJitter(p=.5),
                       # A.ElasticTransform(alpha=1,
                       #                    sigma=30,
                       #                    p=1.),
                       A.PixelDropout(dropout_prob=.001,
                                      p=.5)
                       ]
    if os.path.exists(train_dir):
        visualize_transformations(transformations, train_dir, ncols=3,
                                  masked=True)
    else:
        visualize_transformations(transformations, mini_dataset, ncols=3,
                                  masked=True)


def test_dataset():
    base_dir = mini_dataset
    ds = TeethDataset(base_dir,
                           A.LongestMaxSize(max_size=512),
                           normalize_images=False)
    image, mask, index = ds.get_random()
    masked_image = overlay_mask_on_image(image, mask, denorm=False)
    plt.imshow(to_channel_last(masked_image))
    plt.title(f"Random Example image (index {index})")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    test_dataset()
    test_transformations()
