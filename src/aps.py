import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

# The following code is based on:
# https://github.com/achaman2/truly_shift_invariant_cnns/blob/main/imagenet_exps/models_imagenet/aps_models/apspool.py

ki = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)


class ApsPool(Layer):
    def __init__(self, pool_size: int = 2, pool_stride: int = 2, kernel_size: int = 1, pool_mode: str = 'max',
                 apspool_criterion='l2', **kwargs):

        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.blur_kernel = None
        self.kernel_size = kernel_size
        self.pool_mode = pool_mode
        self.apspool_criterion = apspool_criterion
        super(ApsPool, self).__init__(**kwargs)


    def build(self, input_shape):

        if self.kernel_size > 1:
            # construct blurring filter
            a = construct_1d_array(self.kernel_size)
            filt_np = a[:, None] * a[None, :]
            filt_np /= np.sum(filt_np)
            # this is for channel_last
            filt_np = np.repeat(filt_np[:, :, None], input_shape[-1], -1)

            filt_np = np.expand_dims(filt_np, -1)
            # (3, 3, 128, 1)
            blur_init = tf.keras.initializers.Constant(filt_np)

            if 'learn' in self.pool_mode:
                # softmax without constraint (done in call())
                # softmax constraints all weights to be positive and add up to unity.
                self.blur_kernel = self.add_weight(name='blur_kernel_learn',
                                                   shape=(self.kernel_size, self.kernel_size, input_shape[-1], 1),
                                                   initializer=ki,
                                                   trainable=True)

            else:
                self.blur_kernel = self.add_weight(name='blur_kernel',
                                                   shape=(self.kernel_size, self.kernel_size, input_shape[-1], 1),
                                                   initializer=blur_init,
                                                   trainable=False)

        self.permute_indices = None
        super(ApsPool, self).build(input_shape)


    def call(self, input_to_pool):

        if self.kernel_size > 1 and 'learn' in self.pool_mode:
            # only if we do learnable blurring
            # apply flatten and softmax to each 5x5 filter separately
            # self.blur_kernel is (5, 5, n_filt, 1)  --> (25, n_filt, 1)
            # flatten each filter
            constrained_lpf = tf.reshape(self.blur_kernel, [self.blur_kernel.shape[0] * self.blur_kernel.shape[1], self.blur_kernel.shape[2], 1])
            # softmax to each filter
            constrained_lpf = tf.nn.softmax(constrained_lpf, axis=0, name='softmax_constr')
            # back to squared shape
            low_pass_filter = tf.reshape(constrained_lpf,
                                         [self.blur_kernel.shape[0], self.blur_kernel.shape[1], self.blur_kernel.shape[2], 1])

        else:
            low_pass_filter = self.blur_kernel

        # 1) Apply dense max pooling evaluation (if original pooling is max-pooling)
        if 'avg' not in self.pool_mode and 'Avg' not in self.pool_mode:
            # dense max pooling evaluation, following original pooling size self.pool_size, and unit stride
            inp = tf.nn.pool(input_to_pool,
                           window_shape=(self.pool_size, self.pool_size),
                           strides=(1, 1), padding='SAME', pooling_type='MAX', data_format='NHWC')

        # 2) Apply APS, after the dense max-pool eval
        polyphase_indices = None

        if self.kernel_size == 1:
            # only APS - no blurring
            return aps_downsample_v2(aps_pad(inp),                                      # pad to have even fmap
                                     self.pool_stride,                                  # stride
                                     polyphase_indices,                                 # polyphase_indices = None
                                     permute_indices=self.permute_indices,              # None
                                     apspool_criterion=self.apspool_criterion)          # l2

        else:
            # we do blurring followed by APS

            # 2) Apply blur_kernel but NOT subsample using depthwise_conv2d

            # strides in depthwise_conv performs the subsampling, which is usually 2x2
            # - we first apply low pass (antialiasing) filtering in both time and freq
            # - then we do NOT DO subsampling in both dimensions, specified with strides in depthwise_conv2d
            strides = [1, 1, 1, 1]  # only blurring
            # subsampling done via aps
            blurred_inp = tf.nn.depthwise_conv2d(
                                                inp,
                                                low_pass_filter,   # self.blur_kernel,
                                                strides=strides,
                                                padding='SAME',
                                                data_format='NHWC',
                                                name='antialiasing_subsampling'
            )

            return aps_downsample_v2(aps_pad(blurred_inp),                              # pad to have even fmap
                                     self.pool_stride,                                  # stride
                                     polyphase_indices,                                 # polyphase_indices = None
                                     permute_indices=self.permute_indices,              # None
                                     apspool_criterion=self.apspool_criterion)          # l2


def aps_downsample_v2(x, stride, polyphase_indices=None, permute_indices=None, apspool_criterion='l2'):

    if stride == 1:
        return x

    elif stride > 2:
        raise Exception('Stride>2 currently not supported in this implementation')

    else:
        B, T, F, C = x.shape

        # N_poly is length of flattened fmap after subsampling
        N_poly = int(T*F / 4)
        T_sub = int(T / 2)
        F_sub = int(F / 2)

        if permute_indices is None:
            # permute_indices is the 1-D tensor containing the concat of the 4 possible grids to index the fmap.
            # length is 4* number of elements in subsampled fmap
            permute_indices = permute_polyphase(T, F)

        # reshape by flattening time x freq as the subsampling is going to be done in 1D
        B = tf.shape(x)[0]
        x = tf.reshape(x, [B, -1, C])

        # Downsample the fmap using the 4 possible grids.
        # The subsampling grid is discarding every second bin, hence the outcome is of length N_poly = int(T*F / 4),
        # for each of the 4 candidates.
        x = tf.gather(params=x, indices=permute_indices, axis=-2)
        x = tf.reshape(x, [B, 4, N_poly, C])

        if polyphase_indices is None:
            polyphase_indices = get_polyphase_indices_v2(x, apspool_criterion)
        # polyphase_indices is a vector of batch_size numbers from 0 to 3 eg [0, 3, 2, 3, 1, ...]
        # basically, which candidate fmap is the best, for every example in the batch

        # we have to pair the selection:
        # B index - best fmap
        # 0 - 3
        # 1 - 2
        # 2 - 3
        # 3 - 0
        # ...

        # create index of batch elements to index them
        batch_indices = tf.range(B)
        batch_indices = tf.expand_dims(batch_indices, axis=1)
        best_fmap_per_example = tf.expand_dims(polyphase_indices, axis=1)

        indices_pairs = tf.concat([batch_indices, best_fmap_per_example], axis=1)
        # prepare shape for gather_nd
        indices = tf.expand_dims(indices_pairs, axis=1)
        out_selected_fmap = tf.gather_nd(x, indices)
        # (B, 1, N_poly, C)

        output = tf.reshape(out_selected_fmap, [B, T_sub, F_sub, C])
        # print('output must be (B, T_sub, F_sub, C)', output.shape)

        return output


def get_polyphase_indices_v2(x, apspool_criterion):
    # x has the form [B, 4, N_poly, C] where N_poly corresponds to the reduced version of the 2d feature maps
    # batch, 4 candidate fmaps, flattened fmap of length N_poly = int(T*F / 4) for each of the 4 candidates, channels

    B, Ncandi, N_poly, C = x.shape
    B = tf.shape(x)[0]

    if apspool_criterion == 'l2':
        # l2 norm after flattening dims of interest
        x = tf.reshape(x, [B, Ncandi, N_poly*C])
        norms = tf.norm(x, ord=2, axis=2)
        polyphase_indices = tf.math.argmax(norms, axis=1, output_type=tf.dtypes.int32)

    elif apspool_criterion == 'l1':
        # usually best criterion
        norms = tf.norm(x, ord=1, axis=(2, 3))
        # return index of candidate maximizing criterion
        polyphase_indices = tf.math.argmax(norms, axis=1, output_type=tf.dtypes.int32)

    elif apspool_criterion == 'l_infty':
        B = tf.shape(x)[0]
        # flatten last two dims (the flattened fmap and the depth channels)
        x = tf.reshape(x, [B, 4, -1])
        x = tf.math.abs(x)
        # l_inf: pick max(abs(.))
        max_vals = tf.math.reduce_max(x, axis=2)
        # (B, 4)
        polyphase_indices = tf.math.argmax(max_vals, axis=1, output_type=tf.dtypes.int32)

    elif apspool_criterion == 'euclidean':
        norms = tf.norm(x, ord='euclidean', axis=(2, 3))
        polyphase_indices = tf.math.argmax(norms, axis=1, output_type=tf.dtypes.int32)

    elif apspool_criterion == 'l_infty_tf':
        norms = tf.norm(x, ord=np.inf, axis=(2, 3))
        polyphase_indices = tf.math.argmax(norms, axis=1, output_type=tf.dtypes.int32)

    elif apspool_criterion == 'var':
        # flatten dims of interest
        x = tf.reshape(x, [B, Ncandi, N_poly*C])
        # compute criterion for each candidate (4 variances)
        _, variances = tf.nn.moments(x, axes=[2], name='variance')
        polyphase_indices = tf.math.argmax(variances, axis=1, output_type=tf.dtypes.int32)

    elif apspool_criterion == 'l0':
        # flatten dims of interest
        x = tf.reshape(x, [B, Ncandi, N_poly*C])
        # compute criterion for each candidate (4 L0 norms)
        norms = tf.math.count_nonzero(x, axis=2, dtype=tf.dtypes.int32)
        polyphase_indices = tf.math.argmax(norms, axis=1, output_type=tf.dtypes.int32)

    elif apspool_criterion == 'kurtosis':
        # flatten dims of interest
        x = tf.reshape(x, [B, Ncandi, N_poly*C])
        # compute criterion for each candidate (4 kurtosis values)
        kurtosis = Moment(4, x, standardize=True, reduction_indices=[2])[1] - 3
        # (B, 4)
        polyphase_indices = tf.math.argmax(kurtosis, axis=1, output_type=tf.dtypes.int32)

    elif apspool_criterion == 'skewness': # ok
        # flatten dims of interest
        x = tf.reshape(x, [B, Ncandi, N_poly*C])
        # compute criterion for each candidate (4 skewness values)
        skewness = Moment(3, x, standardize=True, reduction_indices=[2])[1]
        # (B, 4)
        polyphase_indices = tf.math.argmax(skewness, axis=1, output_type=tf.dtypes.int32)

    else:
        raise Exception('Unknown APS criterion')

    return polyphase_indices


def construct_1d_array(filt_size):
    if filt_size == 1:
        a = np.array([1., ])
    elif filt_size == 2:
        a = np.array([1., 1.])
    elif filt_size == 3:
        a = np.array([1., 2., 1.])
    elif filt_size == 4:
        a = np.array([1., 3., 3., 1.])
    elif filt_size == 5:
        a = np.array([1., 4., 6., 4., 1.])
    elif filt_size == 6:
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif filt_size == 7:
        a = np.array([1., 6., 15., 20., 15., 6., 1.])
    return a


def aps_pad(x):
    """
    this func pads in order to have even fmap
    reflect at the end of time and in the HF end, which are presumably less critical locations.
    :param x:
    :return:
    """

    B, T, F, C = x.shape

    if T % 2 == 0 and F % 2 == 0:
        # if even
        return x

    # pad to make even fmap
    if T % 2 != 0:
        x = tf.pad(x, [[0, 0], [0, 1], [0, 0], [0, 0]], mode="REFLECT")

    if F % 2 != 0:
        x = tf.pad(x, [[0, 0], [0, 0], [0, 1], [0, 0]], mode="REFLECT")
    return x


def permute_polyphase(T, F, stride=2):
    """
    given the dims of a squared fmap, T and F, define the 4 possible grids to subsample the fmap with stride 2
    considering a fmap of 2D flattened out to 1D
    """

    # F
    base_even_ind = 2 * tf.range(int(F / 2))[None, :]
    base_odd_ind = 1 + 2 * tf.range(int(F / 2))[None, :]

    # T
    even_increment = 2 * F * tf.range(int(T / 2))[:, None]
    # one index every second time frame. Each index must accumulate the number of freq bands so far (n2F)

    odd_increment = F + 2 * F * tf.range(int(T / 2))[:, None]
    # one index every second time frame. Each index must accumulate the number of freq bands so far (n2F + F)

    p0_indices = tf.reshape(base_even_ind + even_increment, [-1])  # flatten
    p1_indices = tf.reshape(base_even_ind + odd_increment, [-1])  # flatten
    p2_indices = tf.reshape(base_odd_ind + even_increment, [-1])  # flatten
    p3_indices = tf.reshape(base_odd_ind + odd_increment, [-1])  # flatten

    # finally, the 4 possible sets of indices for subsampling incoming fmap in different ways, and flattened to 1D
    permute_indices = tf.concat([p0_indices, p1_indices, p2_indices, p3_indices], axis=0)
    # 1-D tensor containing the concat of the 4 grids to index the fmap.
    # length is 4* number of elements in subsampled fmap
    return permute_indices


def Moment(k, tensor, standardize=False, reduction_indices=None, mask=None):
    """Compute the k-th central moment of a tensor, possibly standardized.
    Args:
    k: Which moment to compute. 1 = mean, 2 = variance, etc.
    tensor: Input tensor.
    standardize: If True, returns the standardized moment, i.e. the central
      moment divided by the n-th power of the standard deviation.
    reduction_indices: Axes to reduce across. If None, reduce to a scalar.
    mask: Mask to apply to tensor.
    Returns:
    The mean and the requested moment.
    """

    # get the divisor
    if reduction_indices is None:
        divisor = tf.constant(np.prod(tensor.get_shape().as_list()), tensor.dtype)
    else:
        divisor = 1.0
        for i in range(len(tensor.get_shape())):
            if i in reduction_indices:
                divisor *= tensor.get_shape()[i]
        divisor = tf.constant(divisor, tensor.dtype)

    # compute the requested central moment
    # note that mean is a raw moment, not a central moment
    mean = tf.math.divide(
        tf.reduce_sum(tensor, axis=reduction_indices, keepdims=True), divisor)
    delta = tensor - mean

    moment = tf.math.divide(tf.reduce_sum(tf.math.pow(delta, k), axis=reduction_indices, keepdims=True), divisor)
    moment = tf.squeeze(moment, reduction_indices)
    if standardize:
        moment = tf.multiply(
            moment,
            tf.math.pow(
                tf.math.rsqrt(Moment(2, tensor, reduction_indices=reduction_indices)[1]),
                k))

    return tf.squeeze(mean, reduction_indices), moment
