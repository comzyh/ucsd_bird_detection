import tensorflow as tf

# https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

################################################################################
# Convenience functions for building the ResNet model.
################################################################################


def batch_norm(inputs, training, data_format):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    return tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=training, fused=True)


def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      data_format: The input format ('channels_last' or 'channels_first').
    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)


def _building_block_v2(inputs, filters, training, projection_shortcut, strides,
                       data_format):
    """A single block for ResNet v2, without a bottleneck.
    Batch normalization then ReLu then convolution as described by:
      Identity Mappings in Deep Residual Networks
      https://arxiv.org/pdf/1603.05027.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the convolutions.
      training: A Boolean for whether the model is in training or inference
        mode. Needed for batch normalization.
      projection_shortcut: The function to use for projection shortcuts
        (typically a 1x1 convolution when downsampling the input).
      strides: The block's stride. If greater than 1, this block will ultimately
        downsample the input.
      data_format: The input format ('channels_last' or 'channels_first').
    Returns:
      The output tensor of the block; shape should match inputs.
    """
    shortcut = inputs
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format)

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=1,
        data_format=data_format)

    return inputs + shortcut


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name, data_format):
    """Creates one layer of blocks for the ResNet model.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the first convolution of the layer.
      bottleneck: Is the block created a bottleneck block.
      block_fn: The block to use within the model, either `building_block` or
        `bottleneck_block`.
      blocks: The number of blocks contained in the layer.
      strides: The stride to use for the first convolution of the layer. If
        greater than 1, this layer will ultimately downsample the input.
      training: Either True or False, whether we are currently training the
        model. Needed for batch norm.
      name: A string name for the tensor output of the block layer.
      data_format: The input format ('channels_last' or 'channels_first').
    Returns:
      The output tensor of the block layer.
    """

    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = filters * 4 if bottleneck else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
            data_format=data_format)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                      data_format)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, training, None, 1, data_format)

    return tf.identity(inputs, name)


def resnet18_v2(inputs, N_final=4, is_training=True, data_format='channels_last'):

    filters = [64, 64, 128, 256, 512]
    num_blocks = [2, 2, 2, 2]
    # input 224 x 224 x 3
    inputs = conv2d_fixed_padding(inputs=inputs,
                                  filters=filters[0],
                                  kernel_size=7,
                                  strides=2,
                                  data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')
    # 112 x 112 x 3

    inputs = tf.layers.max_pooling2d(inputs=inputs,
                                     pool_size=3,
                                     strides=2,
                                     padding='SAME',
                                     data_format=data_format)
    inputs = tf.identity(inputs, 'initial_max_pool')
    # 56 x 56 x 3

    # block layer
    # [3x3x64, 3x3x64] x 2
    inputs = block_layer(inputs=inputs,
                         filters=filters[1],
                         bottleneck=False,
                         block_fn=_building_block_v2,
                         blocks=num_blocks[0],
                         strides=1,
                         training=is_training,
                         name='block_layer_1',
                         data_format=data_format)
    # 56 x 56 x 3

    # [3x3x128, 3x3x128] x 2
    inputs = block_layer(inputs=inputs,
                         filters=filters[2],
                         bottleneck=False,
                         block_fn=_building_block_v2,
                         blocks=num_blocks[1],
                         strides=2,
                         training=is_training,
                         name='block_layer_2',
                         data_format=data_format)
    # 28 x 28 x 3

    # [3x3x256, 3x3x256] x 2
    inputs = block_layer(inputs=inputs,
                         filters=filters[3],
                         bottleneck=False,
                         block_fn=_building_block_v2,
                         blocks=num_blocks[2],
                         strides=2,
                         training=is_training,
                         name='block_layer_3',
                         data_format=data_format)
    # 14 x 14 x 3

    # [3x3x512, 3x3x512] x 2
    inputs = block_layer(inputs=inputs,
                         filters=filters[4],
                         bottleneck=False,
                         block_fn=_building_block_v2,
                         blocks=num_blocks[3],
                         strides=2,
                         training=is_training,
                         name='block_layer_4',
                         data_format=data_format)
    # 7 x 7 x 3
    inputs = batch_norm(inputs, is_training, data_format)
    inputs = tf.nn.leaky_relu(inputs)

    print('line215', inputs)
    inputs = tf.reshape(inputs, [-1, 7 * 7 * filters[4]])
    inputs = tf.layers.dense(inputs=inputs,
                             units=N_final,
                             activation=None,  # linear
                             #kernel_initializer = tf.initializers.truncated_normal(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00001),
                             name='fc1')
    return inputs
