import miscnn
from miscnn import Data_IO
from miscnn.data_loading.interfaces.nifti_io import NIFTI_interface
from miscnn.neural_network.architecture.unet.standard import Architecture
from miscnn.neural_network.metrics import dice_soft
from miscnn.neural_network.metrics import dice_soft_loss
from miscnn.processing.preprocessor import Preprocessor
from miscnn.processing.subfunctions import Clipping, Normalization, Resampling,Disordering

import tensorflow as tf
import argparse
import nibabel as nib
import os

if __name__ == '__main__':
    # input
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, required=True)
    parser_args = parser.parse_args()

    interface = NIFTI_interface(channels=2, classes=2,duplicate=True)
    data_path = parser_args.image_folder
    data_io = Data_IO(interface, data_path, output_path="/home/results/")

    #Purpose:
    #take raw images from ben glocker then clip, normalize and resample to (1,1,1)
    #then achieve on the fly shuffling

    #How to achieve the purpose:
    #create your own subfunction which applies context disordering on the image
    #the subfunctions are in  miscnn.processing.subfunctions

    #resampling is per default to (1,1,1)
    sf = [Clipping(min=-1024, max=3071), Normalization("minmax"),Resampling(),Disordering()]
    pp = Preprocessor(data_io, data_aug=None, batch_size=1, subfunctions=sf, prepare_subfunctions=False,
                      prepare_batches=False, analysis="patchwise-crop", patch_shape=(96,128,128,2))
    # pp = Preprocessor(data_io, data_aug=None, batch_size=1, subfunctions=sf, prepare_subfunctions=False, prepare_batches=False, analysis="patchwise-crop", patch_shape=(80,160,160))

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    # changes according to payer's architecture, constantly 64 filters and no batch_normalization
    unet_standard = Architecture(n_filters=64, batch_normalization=False)
    model = miscnn.Neural_Network(preprocessor=pp, architecture=unet_standard,
                                  loss=tf.keras.losses.BinaryCrossentropy(), metrics=dice_soft)

    sample_list = data_io.get_indiceslist()

    checkpoint_path = "/home/weights/model"
    # options = tf.Checkpoint()
    # save the weights, all without the ones from the Adam and without the first layer
    # save the weights with other names so that they fit the model from payer
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    # save_weights_only=True,
    # verbose=1,
    # options = options)

    for i, layer in enumerate(model.model.layers):
        print(layer.name)
    model.train([sample_list[0]], epochs=3, iterations=10)
    # print(model.model.summary())
    weights = model.model.get_weights()
    for weight in weights:
        print(weight.shape)
    model.model.save_weights(checkpoint_path)