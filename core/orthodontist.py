import os.path

import cv2
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.v2 as transforms
import torch
from PIL import Image

from core.braces import Braces
from core.dataset import TeethDataset
from core.model_inference import TeethPredictor
from core.segmentation import TeethSegmentor


class Orthodontist:
    def __init__(self, model_no=30):
        self.predictor = TeethPredictor(run=model_no)
        self.segmentor = TeethSegmentor()
        self.post_process_config = {
            "num_components": 1,  # todo: whole process not yet for more than 1 components
            "threshold": .2,
            "min_size": 1000
        }
        self.braces_config = {  # todo: should be inside the braces config with setters that check their values
            "rows": 0,
            "bracket_size": 10
        }

    @staticmethod
    def load_image_from_path(image_path):
        # this always needs to be exactly like the model was trained.
        # Should be refactored.
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = A.Normalize(mean=TeethDataset.means, std=TeethDataset.stds)(
            image=image)['image']
        image = ToTensorV2()(image=image)['image']
        image = transforms.ToDtype(torch.float32, scale=True)(image)
        return image

    def change_config(self, new_config):
        for key in new_config:
            if key in list(self.braces_config.keys()):
                try:
                    self.braces_config[key] = int(new_config[key])  # todo: should actually let braces change its config on its own
                except ValueError as e:
                    raise ValueError(f"Provided value {new_config[key]} does not fit into the braces config with key {key}.", e)
            elif key in list(self.segmentor.config.keys()):
                self.segmentor.change_config(key, new_config[key])
            elif new_config[key] in [None, 'undefined']:
                continue
            else:
                raise KeyError(f"Key {key} does not exist in Orthodontist's configs. The segmentation config's keys are:"
                               f"{list(self.segmentor.config.keys())}. The braces config's keys are:"
                               f"{list(self.braces_config.keys())}")

    def fit_braces(self, image_path):
        # todo: check here if the image is in right format
        if os.path.isfile(image_path):
            image = self.load_image_from_path(image_path)
        else:
            raise FileNotFoundError(f"Image was not found at specified location: {image_path}.")
        teeth_prediction = self.predictor.make_prediction(image, size=256)
        # todo: was wenn keine Prediction gefunden wird?

        teeth_prediction.reduce_components(self.post_process_config['num_components'],
                                           self.post_process_config['threshold'],
                                           self.post_process_config['min_size'])
        # todo: wenn num größer, dann auch segmentation result mehrere elemente!
        [segmentation] = self.segmentor.segment_teeth(teeth_prediction)
        segmentation_interims = self.segmentor.interims
        _, centroids = segmentation.find_centroids()
        used_config = segmentation.config

        braces = Braces(segmentation)
        braces.set_rows(self.braces_config['rows'])
        size = self.braces_config['bracket_size']
        braces.place_brackets(size)
        braces.place_wire(plot=False)
        braces_image = braces.visualize_braces_on_image(plot=False)
        used_config['rows'] = self.braces_config['rows']
        used_config['bracket_size'] = self.braces_config['bracket_size']
        segmentation_interims['position_plot'] = braces.position_plot
        braces_image = Image.fromarray(braces_image, 'RGB')
        return braces_image, used_config, segmentation_interims


def test_orthodontist():
    one_row_image = '3b7d41dc-04c1-4b9b-bab7-eada6e20ad84.jpg'
    two_row_image = '00c0bfd5-bf46-4234-9cbf-41f6309d98b3.jpg'
    two_row_image2 = '000d8b15-6885-4b21-9e57-b28d1532145a.jpg'
    image = '1b2ecd0a-5cc9-4d70-b62a-5e93cb9f5477.jpg'
    bearded_guy = '4db2b9d8-04df-4866-ae8a-8076778b938e.jpg'
    image_path = os.path.join('../minimal_test_data/perfect_teeth_dataset/images',
                              one_row_image)
    ortho = Orthodontist()
    #ortho.braces_config['rows'] = 2
    result, _, _ = ortho.fit_braces(image_path)
    plt.figure(figsize=(6, 8))
    plt.tight_layout()
    plt.imshow(result)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    test_orthodontist()
