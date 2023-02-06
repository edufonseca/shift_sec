import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
import numpy as np

ki = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)

# The following code is based on:
# https://github.com/adobe/antialiased-cnns/blob/master/antialiased_cnns/blurpool.py
# https://github.com/vvigilante/antialiased-cnns-keras/blob/master/antialiasing.py


class MaxBlurPooling2D(Layer):
    def __init__(self, pool_size: int = 2, pool_stride: int = 2, kernel_size: int = 3, pool_mode: str = 'max', **kwargs):
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.blur_kernel = None
        self.kernel_size = kernel_size
        self.pool_mode = pool_mode
        super(MaxBlurPooling2D, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.kernel_size == 1 or self.kernel_size == 0:
            a = np.array([1., ])
        elif self.kernel_size == 2:
            a = np.array([1., 1.])
        elif self.kernel_size == 3:
            a = np.array([1., 2., 1.])
        elif self.kernel_size == 4:
            a = np.array([1., 3., 3., 1.])
        elif self.kernel_size == 5:
            a = np.array([1., 4., 6., 4., 1.])
        elif self.kernel_size == 6:
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif self.kernel_size == 7:
            a = np.array([1., 6., 15., 20., 15., 6., 1.])
        else:
            raise ValueError

        filt_np = a[:, None] * a[None, :]
        filt_np /= np.sum(filt_np)
        # this is for channel_last
        filt_np = np.repeat(filt_np[:, :, None], input_shape[-1], -1)

        filt_np = np.expand_dims(filt_np, -1)
        # (3, 3, 128, 1)
        blur_init = tf.keras.initializers.Constant(filt_np)

        # shape required by depthwise_conv2d, which needs
        # a filter tensor of shape [filter_height, filter_width, in_channels, channel_multiplier]
        self.blur_kernel = self.add_weight(name='blur_kernel',
                                           shape=(self.kernel_size, self.kernel_size, input_shape[-1], 1),
                                           initializer=blur_init,
                                           trainable=False)

        super(MaxBlurPooling2D, self).build(input_shape)


    def call(self, x):

        # 1) Apply dense max pooling evaluation (if original pooling is max-pooling)
        if 'avg' not in self.pool_mode and 'Avg' not in self.pool_mode:
            # dense max pooling evaluation, following original pooling size self.pool_size, and unit stride
            x = tf.nn.pool(x,
                           window_shape=(self.pool_size, self.pool_size),
                           strides=(1, 1), padding='SAME', pooling_type='MAX', data_format='NHWC')

        # 2) Apply blur_kernel & subsample using depthwise_conv2d
        # strides in depthwise_conv performs the subsampling, which is usually 2x2
        # - we first apply low pass (antialiasing) filtering in both time and freq
        # - then we do subsampling in both dimensions, as usual, specified with strides in depthwise_conv2d
        # Must have strides[0] = strides[3] = 1. For the most common case of the same horizontal and vertical strides,
        strides = [1, self.pool_stride, self.pool_stride, 1]

        # default is SAME padding, ie zeropadding
        x = tf.nn.depthwise_conv2d(
            x,
            self.blur_kernel,
            strides=strides,
            padding='SAME',
            data_format='NHWC',
            name='antialiasing_subsampling'
        )
        # The output has in_channels * channel_multiplier channels.
        return x


class MaxBlurPooling2D_learn(Layer):
    def __init__(self, pool_size: int = 2, pool_stride: int = 2, kernel_size: int = 3, pool_mode: str = 'max',
                 **kwargs):
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.blur_kernel = None
        self.kernel_size = kernel_size
        self.pool_mode = pool_mode
        super(MaxBlurPooling2D_learn, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.kernel_size == 1 or self.kernel_size == 0:
            a = np.array([1., ])
        elif self.kernel_size == 2:
            a = np.array([1., 1.])
        elif self.kernel_size == 3:
            a = np.array([1., 2., 1.])
        elif self.kernel_size == 4:
            a = np.array([1., 3., 3., 1.])
        elif self.kernel_size == 5:
            a = np.array([1., 4., 6., 4., 1.])
        elif self.kernel_size == 6:
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif self.kernel_size == 7:
            a = np.array([1., 6., 15., 20., 15., 6., 1.])
        else:
            raise ValueError

        filt_np = a[:, None] * a[None, :]
        filt_np /= np.sum(filt_np)
        # this is for channel_last
        filt_np = np.repeat(filt_np[:, :, None], input_shape[-1], -1)

        filt_np = np.expand_dims(filt_np, -1)
        # (3, 3, 128, 1)
        blur_init = tf.keras.initializers.Constant(filt_np)


        # no constraint
        # self.blur_kernel = self.add_weight(name='blur_kernel_learn',
        #                                    shape=(self.kernel_size, self.kernel_size, input_shape[-1], 1),
        #                                    initializer=blur_init,
        #                                    trainable=True)

        # # NonNeg()
        # self.blur_kernel = self.add_weight(name='blur_kernel_learn',
        #                                    shape=(self.kernel_size, self.kernel_size, input_shape[-1], 1),
        #                                    initializer=blur_init,
        #                                    trainable=True,
        #                                    constraint=tf.keras.constraints.NonNeg())
        #

        # Constraining via softmax in call()
        # softmax constrains all weights to be positive and add up to unity, (done in call())

        # init with BP
        # self.blur_kernel = self.add_weight(name='blur_kernel_learn',
        #                                    shape=(self.kernel_size, self.kernel_size, input_shape[-1], 1),
        #                                    initializer=blur_init,
        #                                    trainable=True)

        # init randomly with ki. This works a bit better than init with BP.
        self.blur_kernel = self.add_weight(name='blur_kernel_learn',
                                           shape=(self.kernel_size, self.kernel_size, input_shape[-1], 1),
                                           initializer=ki,
                                           trainable=True)

        super(MaxBlurPooling2D_learn, self).build(input_shape)


    def call(self, x):

        # apply flatten and softmax to each 5x5 filter separately
        # self.blur_kernel is (5, 5, n_filt, 1)  --> (25, n_filt, 1)
        # flatten each filter
        constrained_lpf = tf.reshape(self.blur_kernel, [self.blur_kernel.shape[0] * self.blur_kernel.shape[1], self.blur_kernel.shape[2], 1])
        # softmax to each filter
        constrained_lpf = tf.nn.softmax(constrained_lpf, axis=0, name='softmax_constr')
        # back to squared shape
        constrained_lpf = tf.reshape(constrained_lpf,
                                     [self.blur_kernel.shape[0], self.blur_kernel.shape[1], self.blur_kernel.shape[2], 1])


        # 1) Apply dense max pooling evaluation (if original pooling is max-pooling)
        if 'avg' not in self.pool_mode and 'Avg' not in self.pool_mode:
            # dense max pooling evaluation, following original pooling size self.pool_size, and unit stride
            x = tf.nn.pool(x,
                           window_shape=(self.pool_size, self.pool_size),
                           strides=(1, 1), padding='SAME', pooling_type='MAX', data_format='NHWC')

        # 2) Apply blur_kernel & subsample using depthwise_conv2d
        # strides in depthwise_conv performs the subsampling, which is usually 2x2
        # - we first apply low pass (antialiasing) filtering in both time and freq
        # - then we do subsampling in both dimensions, as usual, specified with strides in depthwise_conv2d
        # Must have strides[0] = strides[3] = 1. For the most common case of the same horizontal and vertical strides,
        strides = [1, self.pool_stride, self.pool_stride, 1]

        # default is SAME padding, ie zeropadding
        x = tf.nn.depthwise_conv2d(
            x,
            constrained_lpf,  # self.blur_kernel,
            strides=strides,
            padding='SAME',
            data_format='NHWC',
            name='antialiasing_subsampling_learn')

        return x
