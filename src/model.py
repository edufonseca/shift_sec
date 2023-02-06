import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, Conv2D, BatchNormalization, Flatten, \
    Activation, Permute, Concatenate, Multiply, Reshape, GlobalAveragePooling2D, GlobalMaxPool2D, \
    MaxPool2D, AveragePooling2D, GlobalAveragePooling1D, GlobalMaxPool1D
from tensorflow.keras import Model, layers
from tensorflow.keras.regularizers import l2

from lpf import MaxBlurPooling2D, MaxBlurPooling2D_learn
from aps import ApsPool


"""
channels last
"""

ki = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
wd = 5e-6


def timefreq_pool(x, pool_size, stride, tf_pool_mode):

    pool_size1 = pool_size
    pool_size2 = pool_size

    # what is more important? temporal structure? freq structure?
    if tf_pool_mode == 'tf_maxmeantime':
        # for a given band, we blur with avg of 2 time frames
        # then pick the max of every second band
        x = AveragePooling2D(pool_size=(pool_size1,1), strides=(stride,1), padding='same')(x)
        x = MaxPool2D(pool_size=(1,pool_size2), strides=(1,stride), padding='same')(x)

    elif tf_pool_mode == 'tf_maxmeanfreq':
        # for a given time frame, we blur with avg of 2 bands
        # then pick the max of every second time frame
        x = AveragePooling2D(pool_size=(1,pool_size1), strides=(1,stride), padding='same')(x)
        x = MaxPool2D(pool_size=(pool_size2,1), strides=(stride,1), padding='same')(x)

    elif tf_pool_mode == 'tf_meanmaxfreq':
        # pick the max of every second band
        # for a given band, we blur with avg of 2 time frames
        x = MaxPool2D(pool_size=(1,pool_size1), strides=(1,stride), padding='same')(x)
        x = AveragePooling2D(pool_size=(pool_size2, 1), strides=(stride, 1), padding='same')(x)

    elif tf_pool_mode == 'tf_meanmaxtime':
        # pick the max of every second time frame
        # for a given time frame, we blur with avg of 2 bands
        x = MaxPool2D(pool_size=(pool_size1, 1), strides=(stride, 1), padding='same')(x)
        x = AveragePooling2D(pool_size=(1, pool_size2), strides=(1, stride), padding='same')(x)

    return x



def pool_layer(x, pool_setup):
    """
    pool_setup[0] can be:
     - int: pooling over squared dimension expressed by single int
     - tuple: pooling over rectangular dimension expressed by tuple

    size = pool_setup[0]
    stride = pool_setup[1]
    mode = pool_setup[2]

    :param x:
    :param pool_setup:
    :return:
    """

    size = pool_setup[0]
    stride = pool_setup[1]
    mode = pool_setup[2]

    if mode is None:
        # bypass
        x = x

    # pooling between the convolutions within each block
    elif mode == 'mp_invar':
        # Intra-block pooling (IBP): provide partial translation invariance w/o dim reduction.
        # used for beginning of backbone
        x = MaxPool2D(pool_size=size, strides=stride, padding='same')(x)
    elif mode == 'ap_invar':
        # same as before, with AveragePooling2D
        x = AveragePooling2D(pool_size=size, strides=stride, padding='same')(x)

    # pooling between the convolutional blocks:
    elif mode == 'mp_size_same_stride':
        # baseline: conventional pooling of size = stride
        x = MaxPool2D(pool_size=size, strides=size, padding='same')(x)
    elif mode == 'ap_size_same_stride':
        # conventional pooling of size = stride
        x = AveragePooling2D(pool_size=size, strides=size, padding='same')(x)
    elif mode in ['tf_maxmeantime', 'tf_maxmeanfreq', 'tf_meanmaxfreq', 'tf_meanmaxtime']:
        # to summarize the final feature map information before the output classifier
        x = timefreq_pool(x, pool_size=size, stride=stride, tf_pool_mode=mode)
    elif mode == 'blur_pool_2D' or mode == 'blur_pool' or mode == 'avg_blur_pool_2D':
        # TODO: meter info en pool_setup
        # Triangle [1, 2, 1]
        # x = MaxBlurPooling2D(pool_size=size, pool_stride=stride, kernel_size=3, pool_mode=mode)(x)

        # [1., 3., 3., 1.]
        # x = MaxBlurPooling2D(pool_size=size, pool_stride=stride, kernel_size=4, pool_mode=mode)(x)

        # Binomial-5 [1., 4., 6., 4., 1.]
        x = MaxBlurPooling2D(pool_size=size, pool_stride=stride, kernel_size=5, pool_mode=mode)(x)

        # [1., 5., 10., 10., 5., 1.]
        # x = MaxBlurPooling2D(pool_size=size, pool_stride=stride, kernel_size=6, pool_mode=mode)(x)

        # [1., 6., 15., 20., 15., 6., 1.]
        # x = MaxBlurPooling2D(pool_size=size, pool_stride=stride, kernel_size=7, pool_mode=mode)(x)

    elif mode == 'blur_pool_2D_learn':
        # aka TLPF
        # TODO: meter info en pool_setup

        # Binomial-5 [1., 4., 6., 4., 1.]
        # x = MaxBlurPooling2D_learn(pool_size=size, pool_stride=stride, kernel_size=3, pool_mode=mode)(x)
        # x = MaxBlurPooling2D_learn(pool_size=size, pool_stride=stride, kernel_size=4, pool_mode=mode)(x)
        x = MaxBlurPooling2D_learn(pool_size=size, pool_stride=stride, kernel_size=5, pool_mode=mode)(x)
        # x = MaxBlurPooling2D_learn(pool_size=size, pool_stride=stride, kernel_size=6, pool_mode=mode)(x)


    elif mode == 'ApsPool' or mode == 'ApsPool_learn':

        # kernel_size = 1     # no blur
        # kernel_size = 3     # combine APS with BlurPool 3x3 (fixed or learn, depending on 'ApsPool' or 'ApsPool_learn')
        # kernel_size = 4     # combine APS with BlurPool 3x3 (fixed or learn, depending on 'ApsPool' or 'ApsPool_learn')
        kernel_size = 5     # combine APS with BlurPool 5x5 (fixed or learn, depending on 'ApsPool' or 'ApsPool_learn')

        # apspool_criterion = 'l_infty'
        apspool_criterion = 'l1'
        # apspool_criterion = 'l2'

        # apspool_criterion = 'euclidean'
        # apspool_criterion = 'l_infty_tf'

        # apspool_criterion = 'var'
        # apspool_criterion = 'l0'
        # apspool_criterion = 'kurtosis'
        # apspool_criterion = 'skewness'
        # apspool_criterion = 'entropy'

        x = ApsPool(pool_size=size, pool_stride=stride, kernel_size=kernel_size, pool_mode=mode,
                    apspool_criterion=apspool_criterion)(x)

    return x


def conv_block(x, n_filt=64, pool1=(3, 1, None), pool2=(2, 2, 'mp_size_same_stride')):
    """
    poolx = (size, stride, pool_mode)
    :param x:
    :param n_filt:
    :param pool1:
    :param pool2:
    :return:
    """

    x = Conv2D(filters=n_filt, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=ki)(x)
    # x = Conv2D(filters=n_filt, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=ki, kernel_regularizer=l2(wd))(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # gives (partial) translation invariance but not dimensionality reduction.
    x = pool_layer(x, pool1)

    x = Conv2D(filters=n_filt, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=ki)(x)
    # x = Conv2D(filters=n_filt, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=ki, kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = pool_layer(x, pool2)

    return x


def backbone(x, plearn):

    # width factor
    k = plearn.get('width_factor')

    # simple frontend
    x = Conv2D(32*k, kernel_size=3, padding='same', use_bias=False, kernel_initializer=ki)(x)
    # x = Conv2D(32*k, kernel_size=3, padding='same', use_bias=False, kernel_initializer=ki, kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # VGG41 - 1.3M and VGG42 - 4.9M
    # conv_block1============================
    x = conv_block(x, n_filt=48 * k, pool1=(plearn.get('trans_inv'), 1, plearn.get('bb_pool_1')), pool2=(2, 2, plearn.get('bb_pool_2')))

    # conv_block2============================
    x = conv_block(x, n_filt=64 * k, pool1=(plearn.get('trans_inv'), 1, plearn.get('bb_pool_1')), pool2=(2, 2, plearn.get('bb_pool_2')))

    # conv_block3============================
    x = conv_block(x, n_filt=128 * k, pool1=(plearn.get('trans_inv'), 1, plearn.get('bb_pool_1')), pool2=(2, 2, plearn.get('bb_pool_2')))

    # conv_block4
    x = conv_block(x, n_filt=256 * k, pool1=(plearn.get('trans_inv'), 1, None), pool2=(2, 2, None))

    # final global pooling
    # ========BASIC POOLINGS=======================
    if plearn.get('global_pooling') == 'gmp':
        x = GlobalMaxPool2D(name='gmp')(x)
    elif plearn.get('global_pooling') == 'gap':
        x = GlobalAveragePooling2D(name='gap')(x)
    elif plearn.get('global_pooling') == 'gapgmp':
        gap = GlobalAveragePooling2D(name='gap')(x)
        gmp = GlobalMaxPool2D(name='gmp')(x)
        # concat in the time dim so that it is already flattened
        x = tf.concat([gap, gmp], 1)
    elif plearn.get('global_pooling') == 'gapgmpstd':
        gap = GlobalAveragePooling2D(name='gap')(x)
        gmp = GlobalMaxPool2D(name='gmp')(x)
        _, v = tf.nn.moments(x, axes=[1, 2], name='variance')
        std = tf.math.sqrt(v + tf.constant(1e-12), name='std')
        # concat in the time dim so that it is already flattened
        x = tf.concat([gap, gmp, std], 1)

    # ========TIME-FREQ POOLINGS=======================
    elif plearn.get('global_pooling') == 'meanmaxtime':
        # max temporal pooling followed by average freq pooling
        # (None, time, freq, channels)
        freq_map = tf.reduce_max(x, axis=1)
        x = GlobalAveragePooling1D()(freq_map)
    elif plearn.get('global_pooling') == 'maxmeanfreq':
        # best so far
        # mean freq pooling followed by max temporal pooling
        # (None, time, freq, channels)
        time_map = tf.reduce_mean(x, axis=2)
        x = GlobalMaxPool1D()(time_map)
    elif plearn.get('global_pooling') == 'meanmaxfreq':
        # max freq pooling followed by mean temporal pooling
        # (None, time, freq, channels)
        time_map = tf.reduce_max(x, axis=2)
        x = GlobalAveragePooling1D()(time_map)
    elif plearn.get('global_pooling') == 'maxmeantime':
        # mean temporal pooling followed by max freq pooling
        # (None, time, freq, channels)
        freq_map = tf.reduce_mean(x, axis=1)
        x = GlobalMaxPool1D()(freq_map)

    return x


def get_model(pextract=None, plearn=None):

    # (None, time, freq, channels=1)
    input_shape = (pextract.get('patch_len'), pextract.get('n_mels'), 1)
    n_class = plearn.get('n_classes')

    x_in = Input(shape=input_shape)
    x = x_in

    x = backbone(x, plearn)

    # backend
    logits = Dense(n_class, name='logits', use_bias=True, kernel_initializer=ki)(x)
    predictions = Activation(name='predictions', activation='sigmoid')(logits)

    _model = Model(inputs=x_in, outputs=predictions)

    return _model