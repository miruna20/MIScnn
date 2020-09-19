import numpy as np
# Internal libraries/scripts
from miscnn.processing.subfunctions.abstract_subfunction import Abstract_Subfunction

#-----------------------------------------------------#
#          Subfunction class: Normalization           #
#-----------------------------------------------------#
""" 

"""
class Scaling(Abstract_Subfunction):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, mode="z-score",scale=1.0):
        self.mode = mode
        self.scale = scale

    #---------------------------------------------#
    #                Preprocessing                #
    #---------------------------------------------#
    def preprocessing(self, sample, training=True):
        # Access image
        image = sample.img_data

        image_scaled = image * self.scale

        # Update the sample with the normalized image
        sample.img_data = image_scaled

    #---------------------------------------------#
    #               Postprocessing                #
    #---------------------------------------------#
    def postprocessing(self, prediction):
        return prediction
