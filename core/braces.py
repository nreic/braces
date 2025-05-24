import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

from core.segmentation import TeethSegmentation


class Braces:
    def __init__(self, segmentation: TeethSegmentation):
        """
        The Braces object combines the original image on which the braces can
        be displayed, the centroids coordinates in that image of the teeth in
        that image and the teeth bbox.

        The braces_mask contains bool information about if there is a braces
        sticker pixel at the specific position. The braces_sticker contains the
        image information if the braces. Both are empty at first, but apply
        the functions: place_brackets() and place_wire() to create braces.

        Display the image including the braces with visualize_braces_on_image()
        which returns the combined image.

        :param segmentation: takes only a TeethSegmentation
        """
        # np array, uint8, rgb, channel-last
        self.original_image = segmentation.original_image
        self.centroid_groups = np.array([segmentation.global_centroids])  # list of x-y-tuples
        self.teeth_bbox = segmentation.bbox  # (x_start, y_start, width, height)
        self.rows = 0
        self.braces_mask = np.full(
            (self.original_image.shape[0], self.original_image.shape[1]),
            False)
        self.braces_sticker = np.zeros(self.original_image.shape,
                                       dtype=np.uint8)
        self.position_plot = None
        self.color = [75, 75, 85]

    def set_rows(self, rows):
        if rows is None:
            return
        try:
            rows = int(rows)
        except ValueError as e:
            raise TypeError(f"Provided value {rows} is not convertable to int.")
        assert rows in [0, 1, 2], f"Number of rows for braces can either be 0, 1, or 2, not {rows}."
        if rows == 2:
            self.centroid_groups = self.separate_centroids(plot=False)
        self.rows = rows

    def cleanup_centroids(self):
        # todo: remove if they are too close together or touching
        #  move or remove if the brackets dont fit on the wire
        pass

    def place_brackets(self, size=10):
        radius = size // 2
        for group in self.centroid_groups:
            for centr_x, centr_y in group:
                for dx in range(-radius, radius+1):
                    for dy in range(-radius, radius+1):
                        x = centr_x + dx
                        y = centr_y + dy
                        if (0 <= x < self.braces_sticker.shape[1] and
                                0 <= y < self.braces_sticker.shape[0]):
                            self.braces_sticker[y, x, :] = self.color
                            self.braces_mask[y, x] = True

        return self.braces_sticker, self.braces_mask

    def separate_centroids(self, y_scale_factor:int=10, plot=True):
        if self.rows == 2:
            raise ValueError("Tried to separate centroids that were already \
            separated.")
        centroids = np.squeeze(self.centroid_groups).copy()
        # stretch centroids in vertical direction to increase clustering along
        # horizontal lines
        centroids[:, 1] *= y_scale_factor
        kmeans = KMeans(n_clusters=2, random_state=0).fit(centroids)
        labels = kmeans.labels_
        cluster_1 = centroids[labels == 0]
        cluster_1 = np.round(cluster_1 / np.array([1, y_scale_factor])).astype(int)
        cluster_2 = centroids[labels == 1]
        cluster_2 = np.round(cluster_2 / np.array([1, y_scale_factor])).astype(int)

        if plot:
            x, y = (np.array([x for x, y in cluster_1]),
                    np.array([y for x, y in cluster_1]))
            plt.scatter(x, y)
            x2, y2 = (np.array([x for x, y in cluster_2]),
                      np.array([y for x, y in cluster_2]))
            plt.scatter(x2, y2, color='r')
            plt.gca().invert_yaxis()
            plt.axis('equal')
            plt.show()
        return [cluster_1, cluster_2]

    def fit_curve(self, data_points):
        x = np.array([x for x, y in data_points]).reshape(-1, 1)
        y = np.array([y for x, y in data_points])

        # Polynomial Regression
        poly = PolynomialFeatures(degree=2)
        x_poly = poly.fit_transform(x)
        reg = LinearRegression()
        reg.fit(x_poly, y)

        # predict the wire
        y_vals = reg.predict(x_poly)
        error = mean_squared_error(y, y_vals)

        return reg, error, x, y, poly

    def place_wire(self, thickness=2, plot=True):
        wires = []
        if self.rows == 0:
            # initial braces placement, assume 1 teeth row
            centroids = np.squeeze(self.centroid_groups).copy()
            reg_1row, error_1row, x, y, poly = self.fit_curve(centroids)

            if error_1row < 10:
                wires.append([reg_1row, x, y, poly])
                self.rows = 1
            else:
                # assume 2 teeth rows
                centroids_2rows = self.separate_centroids(plot=False)
                for row in centroids_2rows:
                    reg_2, _, x2, y2, poly2 = self.fit_curve(row)
                    wires.append([reg_2, x2, y2, poly2])
                self.rows = 2
                self.centroid_groups = centroids_2rows

        elif self.rows in [1, 2]:
            # no initial braces placement, but user defined config
            # centroids are already separated if needed
            for i, row in enumerate(self.centroid_groups):
                reg, _, x, y, poly = self.fit_curve(row)
                wires.append([reg, x, y, poly])

        for wire in wires:
            reg, x, y, poly = wire

            # predict the wire
            min_x, max_x = np.min(x), np.max(x)
            x_vals = np.linspace(min_x, max_x, 100).reshape(-1, 1)
            x_vals_poly = poly.transform(x_vals)
            y_vals = reg.predict(x_vals_poly)

            # update braces sticker and mask
            radius = thickness // 2
            for i, x_val in enumerate(x_vals.flatten()):
                x_int = int(round(x_val))
                y_int = int(round(y_vals[i]))
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if (0 <= x_int + dx < self.braces_sticker.shape[1] and
                                0 <= y_int + dy < self.braces_sticker.shape[0]):
                            self.braces_sticker[y_int + dy, x_int + dx, :] = self.color
                            self.braces_mask[y_int + dy, x_int + dx] = True

            plt.scatter(x, y, label='Brackets')
            plt.plot(x_vals, y_vals, color='r', label='Wire')

        plt.gca().invert_yaxis()
        plt.axis('equal')
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        if plot:
            plt.show()
        else:
            plt.close()
        buf.seek(0)
        braces_plot = Image.open(buf)
        self.position_plot = np.array(braces_plot)

        return self.braces_sticker, self.braces_mask

    def visualize_braces_on_image(self, plot=True):
        combi_image = np.where(self.braces_mask[:, :, None],
                               self.braces_sticker,
                               self.original_image)
        if plot:
            plt.imshow(combi_image)
            plt.axis('off')
            plt.show()
        return combi_image


def test_separation():
    centroids = [(415, 1280), (466, 1288), (592, 1290), (435, 1288),
                 (554, 1293), (623, 1286), (639, 1282), (503, 1297),
                 (624, 1307), (598, 1318), (509, 1327), (538, 1326),
                 (566, 1326)]
    # todo: diese Testmethode ohne Abhängigkeiten zu Orthodontist schreiben!
    pass
