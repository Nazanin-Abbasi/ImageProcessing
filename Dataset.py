import os
from skimage import io
import pandas as pd
import numpy as np


class Dataset:

    def loadData(self):
        images = {}
        for filenames in os.listdir(r"H:\uni\Alzahra/8\Python\HW\05\Data"):
            filename = os.path.join(r"H:\uni\Alzahra\8\Python\HW\05\Data", filenames)
            images[filenames] = io.imread(filename)

        return images

    def setContour(self, contours):
        contourSize = {}
        for image in contours:
            contourSize[image] = len(contours[image])

        return contourSize

    def setCorner(self, corners):
        cornerSize = {}
        for image in corners:
            cornerSize[image] = len(corners[image])

        return cornerSize

    def setConvexHullArea(self, convexHulls):
        convexHullArea = {}
        for image in convexHulls:
            convexHullArea[image] = np.count_nonzero(convexHulls[image])

        return convexHullArea

    def setDataset(self, contourSize, cornerSize, convexHullArea):
        index = list(cornerSize.keys())
        contourSizeList = contourSize.values()
        cornerSizeList = cornerSize.values()
        convexHullAreaList = convexHullArea.values()

        dataset = pd.DataFrame(index=index, columns=['Contour Size', 'Corner Size', 'Convex Hull Area', 'Label'])
        dataset['Contour Size'] = contourSizeList
        dataset['Corner Size'] = cornerSizeList
        dataset['Convex Hull Area'] = convexHullAreaList

        indexes = dataset.index
        counter = 0
        for i in indexes:
            if counter < 50:
                dataset.at[i, 'Label'] = 1
            else:
                dataset.at[i, 'Label'] = 2
            counter = counter + 1

        print(dataset)

        return dataset
