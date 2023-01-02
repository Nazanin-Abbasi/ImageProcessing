from skimage import color, img_as_float, io, exposure
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage.transform import rescale
from MachineLearning import MachineLearning
from Attributes import Attributes
from Dataset import Dataset
import numpy as np
import os


def LoadData():
    images = {}
    for filenames in os.listdir(r"H:\uni\Alzahra\8\Python\HW\05\Photos"):
        filename = os.path.join(r"H:\uni\Alzahra\8\Python\HW\05\Photos", filenames)
        images[filenames] = io.imread(filename)

    return images


def Scaling(images):
    for image in images:
        I = color.rgb2gray(images[image])
        new_image = rescale(I, 0.3)


def Contrast(images):
    for image in images:
        new_image = exposure.equalize_hist(images[image])
        logarithmic_corrected = exposure.adjust_log(images[image])


def Denoising(images):
    for image in images:
        originalImage = img_as_float(images[image])
        sigma_est = np.mean(estimate_sigma(images[image], multichannel=True))
        patch_kw = dict(patch_size=5,
                        patch_distance=6)

        denoise = denoise_nl_means(originalImage, h=0.8 * sigma_est, fast_mode=True, **patch_kw)
        io.imsave(image, denoise)


def Thresholding(images):
    for image in images:
        thresh = threshold_otsu(images[image])
        binary = images[image] > thresh

        thresh_sauvola = threshold_sauvola(images[image])
        binary_sauvola = images[image] > thresh_sauvola


if __name__ == '__main__':
    images = LoadData()
    # Scaling(images)
    # Contrast(images)
    # Denoising(images)
    # Thresholding(images)

    dataset = Dataset()
    data = dataset.loadData()

    attributes = Attributes(data)
    contours = attributes.Contour()
    corners = attributes.Corner()
    convexHulls = attributes.convexHull()

    contourSize = dataset.setContour(contours)
    cornerSize = dataset.setCorner(corners)
    convexHullArea = dataset.setConvexHullArea(convexHulls)
    finalDataset = dataset.setDataset(contourSize, cornerSize, convexHullArea)

    machineLearning = MachineLearning()
    machineLearning.classification(finalDataset)
