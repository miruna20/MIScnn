#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2020 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                     Reference:                      #
#   Olaf Ronneberger, Philipp Fischer, Thomas Brox.   #
#                    18 May 2015.                     #
#          U-Net: Convolutional Networks for          #
#            Biomedical Image Segmentation.           #
#                    MICCAI 2015.                     #
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose, AveragePooling3D, UpSampling3D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
# Internal libraries/scripts
from miscnn.neural_network.architecture.abstract_architecture import Abstract_Architecture
import tensorflow as tf

#-----------------------------------------------------#
#         Architecture class: U-Net Standard          #
#-----------------------------------------------------#
""" The Standard variant of the popular U-Net architecture.

Methods:
    __init__                Object creation function
    create_model_2D:        Creating the 2D U-Net standard model using Keras
    create_model_3D:        Creating the 3D U-Net standard model using Keras
"""
class Architecture(Abstract_Architecture):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, n_filters=32, depth=4, activation='softmax',
                 batch_normalization=True):
        # Parse parameter
        self.n_filters = n_filters
        self.depth = depth
        self.activation = activation
        # Batch normalization settings
        self.ba_norm = batch_normalization
        self.ba_norm_momentum = 0.99

    #---------------------------------------------#
    #               Create 2D Model               #
    #---------------------------------------------#
    def create_model_2D(self, input_shape, n_labels=2):
        # Input layer
        inputs = Input(input_shape)
        # Start the CNN Model chain with adding the inputs as first tensor
        cnn_chain = inputs
        # Cache contracting normalized conv layers
        # for later copy & concatenate links
        contracting_convs = []

        # Contracting Layers
        for i in range(0, self.depth):
            neurons = self.n_filters * 2**i
            cnn_chain, last_conv = contracting_layer_2D(cnn_chain, neurons,
                                                        self.ba_norm,
                                                        self.ba_norm_momentum)
            contracting_convs.append(last_conv)

        # Middle Layer
        neurons = self.n_filters * 2**self.depth
        cnn_chain = middle_layer_2D(cnn_chain, neurons, self.ba_norm,
                                    self.ba_norm_momentum)

        # Expanding Layers
        for i in reversed(range(0, self.depth)):
            neurons = self.n_filters * 2**i
            cnn_chain = expanding_layer_2D(cnn_chain, neurons,
                                           contracting_convs[i], self.ba_norm,
                                           self.ba_norm_momentum)

        # Output Layer
        conv_out = Conv2D(n_labels, (1, 1),
                   activation=self.activation)(cnn_chain)
        # Create Model with associated input and output layers
        model = Model(inputs=[inputs], outputs=[conv_out])
        # Return model
        return model

    #---------------------------------------------#
    #               Create 3D Model               #
    #---------------------------------------------#
    def create_model_3D(self, input_shape, n_labels=2):
        #ToDo change architecture according to payer

        # Input layer
        inputs = Input(input_shape)
        # Start the CNN Model chain with adding the inputs as first tensor
        cnn_chain = inputs
        # Cache contracting normalized conv layers
        # for later copy & concatenate links
        contracting_convs = []

        # Contracting Layers
        for i in range(0, self.depth):
            #ToDo number of filters should remain always 64 (like in payer)
            #neurons = self.n_filters * 2**i
            neurons = self.n_filters

            cnn_chain, last_conv = contracting_layer_3D(cnn_chain, neurons,
                                                        self.ba_norm,
                                                        self.ba_norm_momentum,i)
            contracting_convs.append(last_conv)

        # Middle Layer
        # ToDo number of filters should remain always 64 (like in payer)
        #neurons = self.n_filters * 2**self.depth
        neurons = self.n_filters
        cnn_chain = middle_layer_3D(cnn_chain, neurons, self.ba_norm,
                                    self.ba_norm_momentum,4)
        #because of payer's weird thing??
        cnn_chain = middle_layer2_3D(cnn_chain, neurons, self.ba_norm,
                                    self.ba_norm_momentum,4)

        # Expanding Layers
        for i in reversed(range(0, self.depth)):
            # ToDo number of filters should remain always 64 (like in payer)
            #neurons = self.n_filters * 2**i
            neurons = self.n_filters
            cnn_chain = expanding_layer_3D(cnn_chain, neurons,
                                           contracting_convs[i], self.ba_norm,
                                           self.ba_norm_momentum,i)

        # Output Layer
        #ToDo change back to how it was
        """
        conv_out = Conv3D(n_labels, (1, 1, 1),
                          activation=self.activation, name="net/local/output")(cnn_chain)
        """
        #payer's output layer has 1 feature channel and a 3 3 3 kernel
        conv_out = Conv3D(filters=1, kernel_size=(3,3,3),padding="same",
                  activation="sigmoid",name="net/local/output")(cnn_chain)
        # Create Model with associated input and output layers
        model = Model(inputs=[inputs], outputs=[conv_out])
        # Return model
        return model

#-----------------------------------------------------#
#                   Subroutines 2D                    #
#-----------------------------------------------------#
# Create a contracting layer
def contracting_layer_2D(input, neurons, ba_norm, ba_norm_momentum):
    conv1 = Conv2D(neurons, (3,3), activation='relu', padding='same')(input)
    if ba_norm : conv1 = BatchNormalization(momentum=ba_norm_momentum)(conv1)
    conv2 = Conv2D(neurons, (3,3), activation='relu', padding='same')(conv1)
    if ba_norm : conv2 = BatchNormalization(momentum=ba_norm_momentum)(conv2)
    pool = MaxPooling2D(pool_size=(2, 2))(conv2)
    return pool, conv2

# Create the middle layer between the contracting and expanding layers
def middle_layer_2D(input, neurons, ba_norm, ba_norm_momentum):
    conv_m1 = Conv2D(neurons, (3, 3), activation='relu', padding='same')(input)
    if ba_norm : conv_m1 = BatchNormalization(momentum=ba_norm_momentum)(conv_m1)
    conv_m2 = Conv2D(neurons, (3, 3), activation='relu', padding='same')(conv_m1)
    if ba_norm : conv_m2 = BatchNormalization(momentum=ba_norm_momentum)(conv_m2)
    return conv_m2

# Create an expanding layer
def expanding_layer_2D(input, neurons, concatenate_link, ba_norm,
                       ba_norm_momentum):
    up = concatenate([Conv2DTranspose(neurons, (2, 2), strides=(2, 2),
                     padding='same')(input), concatenate_link], axis=-1)
    conv1 = Conv2D(neurons, (3, 3,), activation='relu', padding='same')(up)
    if ba_norm : conv1 = BatchNormalization(momentum=ba_norm_momentum)(conv1)
    conv2 = Conv2D(neurons, (3, 3), activation='relu', padding='same')(conv1)
    if ba_norm : conv2 = BatchNormalization(momentum=ba_norm_momentum)(conv2)
    return conv2

#-----------------------------------------------------#
#                   Subroutines 3D                    #
#-----------------------------------------------------#
# Create a contracting layer
def contracting_layer_3D(input, neurons, ba_norm, ba_norm_momentum,level):
    """
    if level == 0:
        conv1 = Conv3D(neurons, (3,3,3), activation='relu', padding='same',name = "net/local/contracting/level" + str(level) +"/conv0",input_shape=(1,96,128,128,2))(input)
    else:

    """
    conv1 = Conv3D(neurons, (3, 3, 3), activation='relu', padding='same',name="net/local/contracting/level" + str(level) + "/conv0")(input)

    if ba_norm : conv1 = BatchNormalization(momentum=ba_norm_momentum)(conv1)


    conv2 = Conv3D(neurons, (3,3,3), activation='relu', padding='same',name = "net/local/contracting/level" + str(level) +"/conv1")(conv1)
    if ba_norm : conv2 = BatchNormalization(momentum=ba_norm_momentum)(conv2)
    #pool = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    pool = AveragePooling3D(pool_size=(2,2,2))(conv2)
    return pool, conv2

# Create the middle layer between the contracting and expanding layers
def middle_layer_3D(input, neurons, ba_norm, ba_norm_momentum,level):
    conv_m1 = Conv3D(neurons, (3, 3, 3), activation='relu', padding='same',name = "net/local/contracting/level" + str(level) +"/conv0")(input)
    if ba_norm : conv_m1 = BatchNormalization(momentum=ba_norm_momentum)(conv_m1)

    conv_m2 = Conv3D(neurons, (3, 3, 3), activation='relu', padding='same', name = "net/local/contracting/level" + str(level)+"/conv1")(conv_m1)
    if ba_norm : conv_m2 = BatchNormalization(momentum=ba_norm_momentum)(conv_m2)
    return conv_m2

def middle_layer2_3D(input, neurons, ba_norm, ba_norm_momentum,level):
    conv_m1 = Conv3D(neurons, (3, 3, 3), activation='relu', padding='same',name = "net/local/expanding/level" + str(level) +"/conv0")(input)
    if ba_norm : conv_m1 = BatchNormalization(momentum=ba_norm_momentum)(conv_m1)

    conv_m2 = Conv3D(neurons, (3, 3, 3), activation='relu', padding='same', name = "net/local/expanding/level" + str(level)+"/conv1")(conv_m1)
    if ba_norm : conv_m2 = BatchNormalization(momentum=ba_norm_momentum)(conv_m2)
    return conv_m2

# Create an expanding layer
def expanding_layer_3D(input, neurons, concatenate_link, ba_norm,
                       ba_norm_momentum,level):
    #conv3dtranspose = Conv3DTranspose(neurons, (2, 2, 2), strides=(2, 2, 2),padding='same')(input)
    #up = concatenate([conv3dtranspose, concatenate_link], axis=4)

    #ToDo after up hope to have 128 channels
    #up = concatenate([UpSampling3D()(input), concatenate_link], axis=4)
    up = concatenate([resize_trilinear_payer(input,(2,2,2),data_format='channels_last'), concatenate_link], axis=4)

    conv1 = Conv3D(neurons, (3, 3, 3), activation='relu', padding='same',name="net/local/expanding/level" + str(level) +"/conv0")(up)
    if ba_norm : conv1 = BatchNormalization(momentum=ba_norm_momentum)(conv1)

    conv2 = Conv3D(neurons, (3, 3, 3), activation='relu', padding='same',name = "net/local/expanding/level" + str(level)+"/conv1")(conv1)
    if ba_norm : conv2 = BatchNormalization(momentum=ba_norm_momentum)(conv2)
    return conv2

def resize_trilinear_payer(inputs, factors=None, output_size=None, name=None, data_format='channels_first'):
    """
    Trilinearly resizes an input volume to either a given size of a factor.
    :param inputs: 5D tensor.
    :param output_size: Output size.
    :param factors: Scale factors.
    :param name: Name.
    :param data_format: Data format.
    :return: The resized tensor.
    """
    num_batches, num_channels, [depth, height, width] = get_batch_channel_image_size_payer(inputs, data_format)
    num_batches = 1
    dtype = inputs.dtype
    name = name or 'upsample'
    with tf.name_scope(name):
        if data_format == 'channels_first':
            inputs_channels_last = tf.transpose(inputs, [0, 2, 3, 4, 1])
        else:
            inputs_channels_last = inputs

        if output_size is None:
            output_depth, output_height, output_width = [int(s * f) for s, f in zip([depth, height, width], factors)]
        else:
            output_depth, output_height, output_width = output_size

        # resize y-z
        squeeze_b_x = tf.reshape(inputs_channels_last, [-1, height, width, num_channels])
        resize_b_x = tf.cast(tf.image.resize(squeeze_b_x, [output_height, output_width]), dtype=dtype)
        resume_b_x = tf.reshape(resize_b_x, [num_batches, depth, output_height, output_width, num_channels])

        # resize x
        #   first reorient
        reoriented = tf.transpose(resume_b_x, [0, 3, 2, 1, 4])
        #   squeeze and 2d resize
        squeeze_b_z = tf.reshape(reoriented, [-1, output_height, depth, num_channels])
        resize_b_z = tf.cast(tf.image.resize(squeeze_b_z, [output_height, output_depth]), dtype=dtype)
        resume_b_z = tf.reshape(resize_b_z, [num_batches, output_width, output_height, output_depth, num_channels])

        if data_format == 'channels_first':
            output = tf.transpose(resume_b_z, [0, 4, 3, 2, 1])
        else:
            output = tf.transpose(resume_b_z, [0, 3, 2, 1, 4])
        return output

def get_batch_channel_image_size_payer(inputs, data_format):
    inputs_shape = inputs.get_shape().as_list()
    if data_format == 'channels_first':
        if len(inputs_shape) == 4:
            return inputs_shape[0], inputs_shape[1], inputs_shape[2:4]
        if len(inputs_shape) == 5:
            return inputs_shape[0], inputs_shape[1], inputs_shape[2:5]
    elif data_format == 'channels_last':
        if len(inputs_shape) == 4:
            return inputs_shape[0], inputs_shape[3], inputs_shape[1:3]
        if len(inputs_shape) == 5:
            return inputs_shape[0], inputs_shape[4], inputs_shape[1:4]