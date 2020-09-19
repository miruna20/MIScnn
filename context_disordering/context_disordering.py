import miscnn
from miscnn import Data_IO
from miscnn.data_loading.interfaces.nifti_io import NIFTI_interface
from miscnn.neural_network.architecture.unet.standard import Architecture
from miscnn.neural_network.metrics import dice_soft
from miscnn.neural_network.metrics import dice_soft_loss
from miscnn.processing.preprocessor import Preprocessor
from miscnn.processing.subfunctions import Clipping, Normalization, Resampling, Scaling

import tensorflow as tf
import argparse
import nibabel as nib
import os

#class CustomModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
 #   def on_epoch_end(self, epoch, logs=None):
  #      if self.save_freq == 'epoch':
   #         self._save_model(epoch=epoch, logs=logs)


if __name__ == '__main__':

    #input
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, required=True)
    parser_args = parser.parse_args()

    #inputs
    training = True
    checkpoint_path = "/home/miscnn/weights/model"
    path_weights_to_load = "/home/miscnn/weights/model"
    #path_weights_to_load = "/home/payer/model/model"

    #initial declaration and assignments
    interface = NIFTI_interface(channels=2, classes=2,duplicate=True)
    data_path = parser_args.image_folder
    data_io = Data_IO(interface, data_path,output_path="/home/miscnn/results")

    #ToDo: in the future use the preprocessor functions for resampling
    sf = [Scaling(scale = 1.0/2048.0),Clipping(min=-1.0, max=1.0)]
    pp = Preprocessor(data_io,data_aug=None,batch_size=1,subfunctions=sf,prepare_subfunctions=False,prepare_batches=False,analysis="fullimage")

    #GPU settings
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0],enable=True)

    #changes according to payer's architecture, constantly 64 filters and no batch_normalization, explicitly add the activation function, otherwise it s softmax
    unet_standard = Architecture(n_filters=64,batch_normalization=False,activation=None)
    model = miscnn.Neural_Network(preprocessor=pp, architecture=unet_standard,loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=None)

    sample_list = data_io.get_indiceslist()

    #use callbacks in order to track the best network
    """
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     verbose=1)
    """

    """
    for i,layer in enumerate(model.model.layers):
        print(layer.name)
    """
    if training:
        model.train([sample_list[0]], epochs=100)
        model.model.save_weights(checkpoint_path)
    else:
        model.model.load_weights(path_weights_to_load)

    print(model.model.summary())

    #weights = model.model.get_weights()
    #for weight in weights:
        #print(weight.shape)

    #run inference with the model for which we load the saved weights
    #model.predict([sample_list[0]])

    ##################################
    ###prediction and visualization###
    ##################################

    model.predict([sample_list[1]])
    model.predict([sample_list[0]])

    """# Load the sample
    sample = data_io.sample_loader(sample_list[1], load_seg=True, load_pred=True)
    # Access image, truth and predicted segmentation data
    img, seg, pred = sample.img_data, sample.seg_data, sample.pred_data
    # Visualize the truth and prediction segmentation as a GIF
    from miscnn.utils.visualizer import visualize_evaluation

    # Load the sample
    sample = data_io.sample_loader(sample_list[0], load_seg=True, load_pred=True)
    # Access image, truth and predicted segmentation data
    img, seg, pred = sample.img_data, sample.seg_data, sample.pred_data
    # Visualize the truth and prediction segmentation as a GIF
    from miscnn.utils.visualizer import visualize_evaluation

    visualize_evaluation(sample_list[1], img, seg, pred, "/home")
    """

