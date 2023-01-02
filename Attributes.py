from skimage import measure
from skimage.color import rgb2gray
from skimage.feature import corner_harris, corner_peaks
from skimage.morphology import convex_hull_image


class Attributes:
    def __init__(self, images):
        self.images = images

    def Contour(self):
        contours = {}
        for image in self.images:
            new_image = self.images[image].astype("float32")
            new_image = rgb2gray(new_image)
            contours[image] = measure.find_contours(new_image)

        return contours

    def Corner(self):
        corner = {}
        for image in self.images:
            new_image = self.images[image].astype("float32")
            new_image = rgb2gray(new_image)
            corner[image] = corner_peaks(corner_harris(new_image), min_distance=10, threshold_rel=0.2)

        return corner

    def convexHull(self):
        convexHulls = {}
        for image in self.images:
            new_image = self.images[image].astype("float32")
            new_image = rgb2gray(new_image)
            convexHulls[image] = convex_hull_image(new_image)

        return convexHulls
