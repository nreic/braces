import numpy as np
import matplotlib.pyplot as plt
import cv2

from core.dataset import get_random_samples
from core.paths import mini_dataset
from core.utils import plot_images_as_grid, to_channel_last, use_connected_components, to_uint_np_channel_last
from core.model_inference import TeethPrediction


class TeethSegmentor:
    def __init__(self):
        self.config = {
            "mask_dilation": 0,
            "background_brightness": 255,
            "clahe_gridsize_divisor": 20,
            "clahe_cliplimit": 2.0,
            "brighten_value": 12,
            "thresh_blocksize_fraction": 0.6,
            "closing_kernelsize": (2, 2),
            "closing_iterations": 1,
            "dist_transform_fraction": 0.3,
            "erosion_kernelsize": (7, 3),
            "erosion_iterations": 1
            }
        self.interims = {}

    def change_config(self, key, value):
        if value is None:
            return
        try:
            if key in ["mask_dilation",
                       "background_brightness",
                       "clahe_gridsize_divisor",
                       "brighten_value",
                       "closing_iterations",
                       "erosion_iterations"]:
                value = int(value)
            elif key in ["clahe_cliplimit",
                         "thresh_blocksize_fraction",
                         "dist_transform_fraction"]:
                value = float(value)
            elif key in ["closing_kernelsize",
                         "erosion_kernelsize"]:
                value = tuple(map(int, value.split(',')))
        except TypeError as e:
            raise TypeError(f"Provided value {value} does not fit in the segmentation config of {key}.", e)
        self.config[key] = value

    def crop_to_teeth(self, img:np.ndarray, mask:np.ndarray, bbox, padding=0):
        """
        Crops the image and mask to given bounding box.

        :param bbox: bounding box in x_min, y_min, width, height
        :param padding: increase cropped area on all sides with this padding in px
        """
        x_min, y_min, width, height = bbox
        img_h, img_w = img.shape[0], img.shape[1]

        # adjust with padding, ensure inside of image
        top = int(max(y_min - padding, 0))
        left = int(max(x_min - padding, 0))
        bottom = int(min(y_min + height + padding, img_h))
        right = int(min(x_min + width + padding, img_w))

        cropped_img = img[top:bottom, left:right,:]
        cropped_mask = mask[top:bottom, left:right]

        return cropped_img, cropped_mask, (top, left)

    def black_background(self, image:np.ndarray, mask:np.ndarray, dilate_with_kernel=None, color=None):
        """
        Returns new image colored in color specified in config except for masked area.
        Optional: Dilate mask before coloring with kernel. Mask must be binary for dilation!
        """
        if dilate_with_kernel is not None:
            mask = mask.copy()
            mask = cv2.dilate(mask, dilate_with_kernel, iterations=self.config['mask_dilation'])
        if color == None:
            color = self.config['background_brightness']
        image = image.copy()
        if image.ndim == 2:
            image = np.where(mask[:,:] == 0, color, image)
        elif image.ndim == 3:
            image = np.where(mask[:,:,None] == 0, color, image)
        else:
            raise ValueError("Image dimensions are wrong. Why did work up to this point?")
        return image, mask

    def apply_segmentation_algos(self, image:np.ndarray, mask:np.ndarray, plot=False):
        algo_plots = []
        algo_plots.append({"img": image, "cmap": None,
                           "title": "Teeth Area"})
        self.interims['teeth_area'] = image

        # grayscale
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        algo_plots.append(
            {"img": gray_img.copy(), "title": "Grayscale"})

        # improve contrast with CLAHE
        gridsize = max(2, image.shape[0] // self.config["clahe_gridsize_divisor"])
        clahe = cv2.createCLAHE(clipLimit=self.config["clahe_cliplimit"],
                                tileGridSize=(gridsize, gridsize))
        clahe_img = clahe.apply(gray_img)
        algo_plots.append({"img": clahe_img,
                                    "title": f"CLAHE (clip lim = 2.0, gridsize = {gridsize})"})
        self.interims['clahe'] = clahe_img

        # brighten CLAHE
        brighten_value = self.config["brighten_value"]
        bright_clahe = np.where((255 - clahe_img) < brighten_value,
                                255,
                                clahe_img + brighten_value)
        bright_clahe = cv2.GaussianBlur(bright_clahe, (3, 3),
                                        cv2.BORDER_DEFAULT)
        algo_plots.append(
            {"img": bright_clahe,
             "title": f"Over-saturation (+ {brighten_value})"})
        self.interims['brighten'] = bright_clahe.copy()

        # regional adaptive threshold
        block_size = int(image.shape[0] * self.config["thresh_blocksize_fraction"])
        if block_size % 2 == 0: block_size += 1  # block-size wants to be odd ¯\_(ツ)_/¯
        adapt_thresh_cla = cv2.adaptiveThreshold(bright_clahe, 255,
                                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, block_size,
                                                 0)
        adapt_thresh_cla, _ = self.black_background(adapt_thresh_cla, mask, None, 0)
        algo_plots.append({"img": adapt_thresh_cla.copy(),
                                    "title": "Regional Adaptive Thresholds"})
        self.interims['adapt_thresh'] = adapt_thresh_cla

        # closing for noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           self.config["closing_kernelsize"])
        closing_cla = cv2.morphologyEx(adapt_thresh_cla,
                                       cv2.MORPH_CLOSE, kernel,
                                       iterations=self.config["closing_iterations"])
        algo_plots.append({"img": closing_cla, "title": "Closing"})
        self.interims['closing'] = closing_cla

        # watershed background
        bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sure_bg = cv2.dilate(closing_cla, bg_kernel)
        # algo_plots.append({"img": sure_bg,
        #                    "title": "Sure Background (Dilate)"})

        # distance transform
        dist_cla = cv2.distanceTransform(closing_cla, cv2.DIST_L2, 3)
        algo_plots.append({"img": dist_cla,
                                    "title": "Distance Transform"})


        # watershed foreground and markers
        _, sure_fg = cv2.threshold(dist_cla,
                                   self.config["dist_transform_fraction"] * dist_cla.max(),
                                   255, 0)
        self.interims['dist_transform'] = sure_fg.copy()

        sure_fg = np.uint8(sure_fg)
        sure_fg = cv2.erode(sure_fg, np.ones(self.config["erosion_kernelsize"]),
                            iterations=self.config["erosion_iterations"])
        self.interims['erosion'] = sure_fg.copy()
        # algo_plots.append({"img": sure_fg,
        #                    "title": "Sure FG (Glob.Thr. of Dist)"})
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1  # make background 1 instead of 0
        markers[unknown == 255] = 0
        algo_plots.append({"img": markers.copy(),
                            "cmap": 'jet',
                            "title": "Markers"})
        ws_img = image.copy()
        # todo: watershed on gray image? or on bright-clahe image!
        markers = cv2.watershed(ws_img, markers)
        algo_plots.append({"img": markers,
                            "cmap": 'jet',
                            "title": "Watershed Markers"})
        ws_img[markers == -1] = [255, 0, 0]  # mark the borders red
        algo_plots.append(
            {"img": ws_img,
             "cmap": None,
             "title": "Watershed"})
        self.interims['watershed'] = ws_img

        if plot: plot_images_as_grid(algo_plots)

        return markers

    def segment_teeth(self, image, components=None) -> list['TeethSegmentation']:
        """
        Returns a list of TeethSegmentations each one resulted from the application
        of the segmentation algorithms.
        :param image: TeethPrediction object that contains the original image and its components. Or is the original image itself and then needs components passed additionally.
        :param components: need only be passed if image object is not a TeethPrediction!
        """
        if components is None:
            if isinstance(image, TeethPrediction):
                np_image = to_uint_np_channel_last(image.original_image)
                components = image.components
            else:
                raise ValueError("Object passed to 'segment_teeth' function is not of type TeethPrediction.")
        else:
            np_image = image

        segmented_teeth = []
        for component in components:
            mask = component[3]
            bbox = component[0]
            cropped_image, cropped_mask, top_left_offset = self.crop_to_teeth(np_image,
                                                                        mask,
                                                                        bbox,
                                                                        padding=8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cropped_image, cropped_mask = self.black_background(cropped_image, cropped_mask,
                                                           dilate_with_kernel=kernel)

            markers = self.apply_segmentation_algos(cropped_image, cropped_mask, plot=False)
            segmentation = TeethSegmentation(np_image, self.config, markers, bbox, top_left_offset)
            segmented_teeth.append(segmentation)
        return segmented_teeth


class TeethSegmentation:
    def __init__(self, original_image:np.ndarray, config:dict, markers:np.ndarray, bbox, offset:tuple[int,int]):
        """
        A TeethSegmentation object is the result from the segmentation pipeline
        on **one single** teeth area in the image.

        It combines the original image, the configuration parameters to get
        this segmentation and the corresponding markers, qlso the bbox and the
        offset to the original image, of the teeth area. You can compute the
        centroids of the markers to position brackets.

        Create a TeethSegmentation instance by passing the original image as
        **channel-last np.ndarray**, a segmentation config, the watershed
        markers, bbox and offset of bbox wrt original image.
        """
        self.original_image = original_image
        self.config = config
        self.markers = markers
        self.bbox = bbox
        self.offset = offset
        self.local_centroids:list[tuple[int, int]] = None
        self.global_centroids:list = None

    def find_centroids(self, background=1, borders=-1):
        """ Calculates centroids of segmentation markers using its moments. Returns
        both local and global centroids. """
        centroids = []
        for marker in np.unique(self.markers):
            if marker == background or marker == borders:
                continue
            tooth_map = (self.markers == marker).astype(np.uint8)
            moments = cv2.moments(tooth_map)
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            centroids.append((center_x, center_y))
        self.local_centroids = centroids
        self.global_centroids = self.convert_centroids_to_global_coords()
        return self.local_centroids, self.global_centroids

    def convert_centroids_to_global_coords(self):
        top, left = self.offset
        orig_img_h = self.original_image.shape[0]
        orig_img_w = self.original_image.shape[1]
        global_centroids = []
        for x, y in self.local_centroids:
            global_y = y + top
            global_x = x + left
            global_y = int(min(global_y, orig_img_h - 1))
            global_x = int(min(global_x, orig_img_w - 1))
            global_centroids.append((global_x, global_y))
        return global_centroids

    def visualize_centroids(self, plot=True):
        if self.global_centroids is None:
            raise ValueError("This segmentation does not yet have centroids, please call 'find_centroids' first before visualizing.")
        size = 4
        img = self.original_image.copy()
        for centroid in self.global_centroids:
            y,x = centroid
            for xs in range(max(x-size,0), min(x+size, img.shape[0])):
                for ys in range(max(y-size, 0), min(y+size, img.shape[1])):
                    img[xs, ys, 0] = 1
                    img[xs, ys, 1] = 0
                    img[xs, ys, 2] = 0
        if plot:
            plt.imshow(img)
            plt.axis('off')
            plt.show()
        return img


def test_segmentation_pipeline() -> TeethSegmentation:
    # select random sample
    [sample] = get_random_samples(1, mini_dataset)
    orig_img = to_uint_np_channel_last(sample[0]) # now uint np array, rgb, channel-last!!
    binary = sample[1].squeeze(0).numpy().astype(np.uint8)

    # find the components of the mask
    component = use_connected_components(binary)[0]

    # segment the components
    segmentor = TeethSegmentor()
    [segmentation] = segmentor.segment_teeth(orig_img,[component])
    _, centroids = segmentation.find_centroids()

    segmentation.visualize_centroids()
    return segmentation


if __name__ == '__main__':
    test_segmentation_pipeline()
