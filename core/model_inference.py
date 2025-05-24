import os
import numpy as np
import torch
import torchvision.utils
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as TF

from core.paths import get_model_path, mini_dataset, get_best_models_path
from core.utils import use_connected_components, to_channel_last, plot_images_as_grid
from core.dataset import TeethDataset, denormalize, overlay_mask_on_image
from core.model import UNet

class TeethPredictor:
    def __init__(self, run):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] used {self.device} device.")
        self.model = self.load_model(run)

    def load_model(self, run):
        unet = UNet()
        if run < 30:
            path = get_model_path(run)
        if run == 30:
            path = get_best_models_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"The model path {path} for run {run} does not exist. Please provide the correct run number.")
        if run < 20:
            unet.load_state_dict(torch.load(path))
        elif run >= 20:
            state = torch.load(path, map_location=torch.device(self.device))
            model_state = state.get('state_dict')
            if model_state is None: raise ValueError(
                f"Model's state dict of run {run} does not contain the 'state_dict' key to load the model's parameters.")
            unet.load_state_dict(model_state)
        self.model = unet
        unet = unet.to(self.device)
        unet.eval()
        return unet

    def make_prediction(self, image, size=128):
        orig_image = image.clone()
        orig_size = min(image.shape[-2:])
        image = transforms.Resize(size, antialias=False)(image)
        image = image.to(self.device)
        with (torch.no_grad()):
            prediction = self.model(image.unsqueeze(0))  # add batch dimension
            prediction = torch.sigmoid(prediction)
            prediction = transforms.Resize(orig_size,
                                           interpolation=TF.InterpolationMode.BILINEAR,
                                           antialias=False
                                           )(prediction)
            prediction = prediction.detach().squeeze().cpu()
            return TeethPrediction(orig_image, prediction)


class TeethPrediction:
    def __init__(self, image, prediction, threshold=0.1):
        self.original_image = denormalize(image) # channel-first tensor, denormalized
        self.raw = prediction
        self.binary, self.components = self.apply_threshold(threshold)

    def apply_threshold(self, threshold):
        """ Applies the provided threshold on the raw prediction and updates
        instance attribute 'binary' and 'components' accordingly. """
        self.binary = (self.raw >= threshold).numpy().astype(np.uint8)
        self.components = self.separate_components()
        return self.binary, self.components

    def separate_components(self):
        """ returns a list of bbox, area, centroid and mask """
        self.components = use_connected_components(self.binary)
        print(f"Number of components in prediction: {len(self.components)}.")
        return self.components

    def has_components(self):
        return len(self.components) != 0

    def reset(self):
        self.apply_threshold(0.1)

    def reduce_components(self, num, threshold, min_size):
        """ reduces the components in the prediction to given number and returns
        remaining components that can be considered teeth areas. """
        if len(self.components) <= num:
            print(f"Nothing to reduce. This prediction only has {len(self.components)} components.")
            return self.components
        for func, args in [(self.remove_small_blobs, min_size),
                            (self.apply_threshold, threshold),
                            (self.remove_small_blobs, min_size),
                            (self.keep_largest_components, num)]:
            func(args)
            if len(self.components) <= num:
                break
        return self.components

    def remove_small_blobs(self, min_size=20):
        """ removes all components that are smaller than min_size """
        if not self.has_components():
            print(f"Nothing to remove. This prediction has no components.")
            return
        count = 0
        for component in self.components:
            area = component[1]
            if area < min_size:
                self.components.remove(component)
                count += 1
        print(f"{count} component" + ("s were" if count != 1 else " was"), f"deleted. {len(self.components)} components left.")

    def keep_largest_components(self, num):
        """ removes all but the largest num components """
        if not self.has_components() or num <= 0:
            print(f"This prediction has no components (or did you want to get 0 components?)")
            return []
        largest_comps = sorted(self.components,
                               key=lambda component: component[1],
                               reverse=True)[:num]
        # if components list is too large, this could take a while. There are
        # more efficient algorithms to do this. No need for this yet.
        self.components = largest_comps
        return self.components

    def visualize_components(self, image=None, color='red'):
        if image is None:
            image = self.original_image
        bboxes = []
        for component in self.components:
            x_start, y_start, width, height = component[0]
            bbox = [x_start, y_start, x_start+width, y_start+height]
            bboxes.append(bbox)
        bboxes = torch.tensor(bboxes, dtype=torch.int32)
        image = transforms.ToDtype(torch.uint8, scale=True)(image)
        bboxed_image = torchvision.utils.draw_bounding_boxes(image, bboxes, colors=color, width=3)
        return bboxed_image

def test_prediction_pipeline(run, num_images=2):
    ds = TeethDataset(mini_dataset)
    predictor = TeethPredictor(run=run)
    plots = []

    # iterate over the randomly selected images
    for i in range(num_images):
        image, mask, index = ds.get_random() # ds[25]
        masked_image = overlay_mask_on_image(image, mask, denorm=True)
        plots.append({"img": to_channel_last(masked_image.clone()),
                            "title": f"Ground Truth (Image {index})"})

        prediction = predictor.make_prediction(image, size=256)
        plots.append({"img": prediction.raw.numpy().copy(),
                            "cmap": 'gray',
                            "title": "Raw Prediction"})

        bboxed_image = prediction.visualize_components(color='blue')
        # select best component from the prediction
        prediction.reduce_components(1,.1, 1000)

        bboxed_image = prediction.visualize_components(bboxed_image)
        plots.append({"img": to_channel_last(bboxed_image),
                            "title": "Post-Processed Prediction"})

    plot_images_as_grid(plots, ncols=3, figsize=(20, 20))


if __name__ == '__main__':
    test_prediction_pipeline(run=30, num_images=4)