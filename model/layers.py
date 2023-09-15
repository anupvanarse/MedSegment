import tensorflow as tf
import keras_core
import keras_core.ops as ops

def _crop_and_concat(inputs, residual_input):
    
    """ Perform a central crop of ``residual_input`` and concatenate to ``inputs``.
    Args:
        inputs (tf.Tensor): Tensor with input
        residual_input (tf.Tensor): Residual input

    Return:
        Concatenated tf.Tensor with the size of ``inputs``

    """
    factor = inputs.shape[1] / residual_input.shape[1]
    return ops.concatenate([inputs, tf.image.central_crop(residual_input, factor)], axis=-1)
    # TODO: replace tf.image.central_crop to keras_core.layers.CenterCrop layer

class InputBlock(keras_core.Model):
    """ Input block for UNet architecture.
    Args:
        filters (int): Number of filters for the convolutional layers.
    """
    def __init__(self, filters):
       
        super().__init__(self)
        
        self.conv1 = keras_core.layers.Conv2D(filters=filters,
                                              kernel_size= (3,3),
                                              activation=keras_core.activations.relu)
        self.conv2 = keras_core.layers.Conv2D(filters=filters,
                                              kernel_size= (3,3),
                                              activation=keras_core.activations.relu)
        self.maxpool = keras_core.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))

    def call(self, inputs):

        out = self.conv1(inputs)
        out = self.conv2(out)
        mp = self.maxpool(out)
        return out, mp

class DownsampleBlock(keras_core.Model):
    """ Downsample block for UNet architecture.
    Args:
        filters (int): Number of filters for the convolutional layers.
    """
    def __init__(self, filters):
        
        super().__init__(self)
        
        self.conv1 = keras_core.layers.Conv2D(filters=filters,
                                              kernel_size= (3,3),
                                              activation=keras_core.activations.relu)
        self.conv2 = keras_core.layers.Conv2D(filters=filters,
                                              kernel_size= (3,3),
                                              activation=keras_core.activations.relu)
        self.maxpool = keras_core.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))

    def call(self, inputs):

        out = self.conv1(inputs)
        out = self.conv2(out)
        mp = self.maxpool(out)
        return out, mp
    
class BottleneckBlock(keras_core.Model):
    """ Bottleneck block for UNet architecture. This is the central block of the UNet.
    Perform two unpadded convolutions with a specified number of filters and upsample
    including dropout before upsampling for training

    Args:
        filters (int): Number of filters for the convolutional layers.
    """
    def __init__(self, filters):
        
        super().__init__(self)
        
        self.conv1 = keras_core.layers.Conv2D(filters=filters,
                                              kernel_size= (3,3),
                                              activation=keras_core.activations.relu)
        self.conv2 = keras_core.layers.Conv2D(filters=filters,
                                              kernel_size= (3,3),
                                              activation=keras_core.activations.relu)
        self.dropout = keras_core.layers.Dropout(rate=0.5)
        self.conv_transpose = keras_core.layers.Conv2DTranspose(filters= filters//2,
                                                                kernel_size=(3,3),
                                                                strides=(2,2),
                                                                padding='same',
                                                                activation=keras_core.activations.relu)
    
    def call(self, inputs, training):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.dropout(out, training=training)
        out = self.conv_transpose(out)
        return out

class UpsampleBlock(keras_core.Model):
    """ Upsample block for UNet architecture.
    Perform two unpadded convolutions with a specified number of filters and upsample
    
    Args:
        filters (int): Number of filters for the convolutional layers.
    """
    def __init__(self, filters, idx):
        
        super().__init__(self)

        self.conv1 = keras_core.layers.Conv2D(filters=filters,
                                              kernel_size= (3,3),
                                              activation=keras_core.activations.relu)
        self.conv2 = keras_core.layers.Conv2D(filters=filters,
                                              kernel_size= (3,3),
                                              activation=keras_core.activations.relu)
        self.conv_transpose = keras_core.layers.Conv2DTranspose(filters= filters//2,
                                                                kernel_size=(3,3),
                                                                strides=(2,2),
                                                                padding='same',
                                                                activation=keras_core.activations.relu)

    def call(self, inputs, residual_input):
        out = _crop_and_concat(inputs, residual_input)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv_transpose(out)
        return out

class OutputBlock(keras_core.Model):
    """ Output block for UNet architecture.
    
    Args:
        filters (int): Number of filters for the convolutional layers.
        n_classes (int): Number of classes to predict.
    """ 
    def __init__(self, filters, n_classes):
        
        super().__init__(self)
        
        self.conv1 = keras_core.layers.Conv2D(filters=filters,
                                              kernel_size= (3,3),
                                              activation=keras_core.activations.relu)
        self.conv2 = keras_core.layers.Conv2D(filters=filters,
                                              kernel_size= (3,3),
                                              activation=keras_core.activations.relu)
        self.conv3 = keras_core.layers.Conv2D(filters=n_classes,
                                              kernel_size= (1,1),
                                              activation=None)
    
    def call(self, inputs, residual_input):
        out = _crop_and_concat(inputs, residual_input)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

