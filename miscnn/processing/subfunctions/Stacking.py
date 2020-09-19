"""
A stacking Subfunction class which receives an image as input and stacks the image with itself
in order to obtain an extra dimension in the numpy array
e.g image of (96,128,128) will be transformed into (96,128,128,2)

"""

# External libraries
import numpy as np
# Internal libraries/scripts
from miscnn.processing.subfunctions.abstract_subfunction import Abstract_Subfunction

class Stacking(Abstract_Subfunction):
    def __init__(self,mode="double"):
        self.mode = mode
    def preprocessing(self, sample, training=True):
        # Access image
        image = sample.img_data
        image2 = sample.img_data
        image = np.reshape(image, [96, 128, 128, 1])
        image2 = np.reshape(image2, [96, 128, 128, 1])
        newimage = np.concatenate((image, image2), axis=3)
        return newimage
