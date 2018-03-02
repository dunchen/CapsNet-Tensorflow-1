from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./tmp/data/", one_hot=True)

py_all = all

# Training Parameters
learning_rate = 0.001
num_steps = 150000
batch_size = 128
display_step = 1000
IMG_SIZE = 28

# Network Parameters
num_input = 784  # Fashion_MNIST data input (img shape: 28*28)
num_classes = 10  # Fashion_MNIST total classes
dense_size = 128
lc_kernel_size = 15
lc_out_channel = 32
lc_stride = 1
lc_out_h = int((IMG_SIZE - lc_kernel_size) / lc_stride) + 1
lc_out_w = int((IMG_SIZE - lc_kernel_size) / lc_stride) + 1
lc_out_num = lc_out_h * lc_out_w
lc_out_size = lc_out_h

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])


def expand_dims(x, axis=-1):
    """Adds a 1-sized dimension at index "axis".
    # Arguments
        x: A tensor or variable.
        axis: Position where to add a new axis.
    # Returns
        A tensor with expanded dimensions.
    """
    return tf.expand_dims(x, axis)


def ndim(x):
    """Returns the number of axes in a tensor, as an integer.
    # Arguments
        x: Tensor or variable.
    # Returns
        Integer (scalar), number of axes.
    # Examples
    """
    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None


def batch_dot(x, y, axes=None):
    """Batchwise dot product.
    `batch_dot` is used to compute dot product of `x` and `y` when
    `x` and `y` are data in batch, i.e. in a shape of
    `(batch_size, :)`.
    `batch_dot` results in a tensor or variable with less dimensions
    than the input. If the number of dimensions is reduced to 1,
    we use `expand_dims` to make sure that ndim is at least 2.
    # Arguments
        x: Keras tensor or variable with `ndim >= 2`.
        y: Keras tensor or variable with `ndim >= 2`.
        axes: list of (or single) int with target dimensions.
            The lengths of `axes[0]` and `axes[1]` should be the same.
    # Returns
        A tensor with shape equal to the concatenation of `x`'s shape
        (less the dimension that was summed over) and `y`'s shape
        (less the batch dimension and the dimension that was summed over).
        If the final rank is 1, we reshape it to `(batch_size, 1)`.
    """
    if isinstance(axes, int):
        axes = (axes, axes)
    x_ndim = ndim(x)
    y_ndim = ndim(y)
    if x_ndim > y_ndim:
        diff = x_ndim - y_ndim
        y = tf.reshape(y, tf.concat([tf.shape(y), [1] * (diff)], axis=0))
    elif y_ndim > x_ndim:
        diff = y_ndim - x_ndim
        x = tf.reshape(x, tf.concat([tf.shape(x), [1] * (diff)], axis=0))
    else:
        diff = 0
    if ndim(x) == 2 and ndim(y) == 2:
        if axes[0] == axes[1]:
            out = tf.reduce_sum(tf.multiply(x, y), axes[0])
        else:
            out = tf.reduce_sum(tf.multiply(tf.transpose(x, [1, 0]), y),
                                axes[1])
    else:
        if axes is not None:
            adj_x = None if axes[0] == ndim(x) - 1 else True
            adj_y = True if axes[1] == ndim(y) - 1 else None
        else:
            adj_x = None
            adj_y = None
        out = tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
    if diff:
        if x_ndim > y_ndim:
            idx = x_ndim + y_ndim - 3
        else:
            idx = x_ndim - 1
        out = tf.squeeze(out, list(range(idx, idx + diff)))
    if ndim(out) == 1:
        out = expand_dims(out, 1)
    return out


def reshape(x, shape):
    """Reshapes a tensor to the specified shape.
    # Arguments
        x: Tensor or variable.
        shape: Target shape tuple.
    # Returns
        A tensor.
    """
    return tf.reshape(x, shape)


def is_sparse(tensor):
    """Returns whether a tensor is a sparse tensor.
    # Arguments
        tensor: A tensor instance.
    # Returns
        A boolean.
    # Example
    ```python
        >>> from keras import backend as K
        >>> a = K.placeholder((2, 2), sparse=False)
        >>> print(K.is_sparse(a))
        False
        >>> b = K.placeholder((2, 2), sparse=True)
        >>> print(K.is_sparse(b))
        True
    ```
    """
    return isinstance(tensor, tf.SparseTensor)


def to_dense(tensor):
    """Converts a sparse tensor into a dense tensor and returns it.
    # Arguments
        tensor: A tensor instance (potentially sparse).
    # Returns
        A dense tensor.
    # Examples
    ```python
        >>> from keras import backend as K
        >>> b = K.placeholder((2, 2), sparse=True)
        >>> print(K.is_sparse(b))
        True
        >>> c = K.to_dense(b)
        >>> print(K.is_sparse(c))
        False
    ```
    """
    if is_sparse(tensor):
        return tf.sparse_tensor_to_dense(tensor)
    else:
        return tensor


def int_shape(x):
    """Returns the shape of tensor or variable as a tuple of int or None entries.
    # Arguments
        x: Tensor or variable.
    # Returns
        A tuple of integers (or None entries).
    # Examples
    ```python
        >>> from keras import backend as K
        >>> inputs = K.placeholder(shape=(2, 4, 5))
        >>> K.int_shape(inputs)
        (2, 4, 5)
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> K.int_shape(kvar)
        (2, 2)
    ```
    """
    if hasattr(x, '_keras_shape'):
        return x._keras_shape
    try:
        return tuple(x.get_shape().as_list())
    except ValueError:
        return None


def concatenate(tensors, axis=-1):
    """Concatenates a list of tensors alongside the specified axis.
    # Arguments
        tensors: list of tensors to concatenate.
        axis: concatenation axis.
    # Returns
        A tensor.
    """
    if axis < 0:
        rank = ndim(tensors[0])
        if rank:
            axis %= rank
        else:
            axis = 0

    if py_all([is_sparse(x) for x in tensors]):
        return tf.sparse_concat(axis, tensors)
    else:
        return tf.concat([to_dense(x) for x in tensors], axis)


# Create local_conv2d wrapper
def local_conv2d(inputs, kernel, kernel_size, strides, output_shape,
                 output_channel):
    """Apply 2D conv with un-shared weights.
    # Arguments
        channels_first ALL!!
        inputs: 4D tensor with shape:
                (batch_size, filters, new_rows, new_cols)
                if data_format='channels_first'
        kernel: the unshared weight for convolution,
                with shape (output_items, feature_dim, filters)
        kernel_size: a tuple of 2 integers, specifying the
                     width and height of the 2D convolution window.
        strides: a tuple of 2 integers, specifying the strides
                 of the convolution along the width and height.
        output_shape: a tuple with (output_row, output_col)
        data_format: the data format, channels_first or channels_last
    # Returns
        A 4d tensor with shape:
        (batch_size, filters, new_rows, new_cols)
        if data_format='channels_first'
    """

    stride_row, stride_col = strides
    output_row, output_col = output_shape
    kernel_shape = int_shape(kernel)
    _, feature_dim, _ = kernel_shape

    xs = []
    for i in range(output_row):
        for j in range(output_col):
            slice_row = slice(i * stride_row,
                              i * stride_row + kernel_size[0])
            slice_col = slice(j * stride_col,
                              j * stride_col + kernel_size[1])
            xs.append(reshape(inputs[:, slice_row, slice_col, :],
                      (1, -1, feature_dim)))

    x_aggregate = concatenate(xs, axis=0)
    output = batch_dot(x_aggregate, kernel)
    output = tf.reshape(output,
                        (output_row, output_col, -1, output_channel))
    output = tf.transpose(output, (2, 0, 1, 3))  # Channels-last
    return output


# Create model
class CapsLayer(object):
    ''' Capsule layer.
    Args:
        input: A 4-D tensor.
        num_outputs: the number of capsule in this layer.
        vec_len: integer, the length of the output vector of a capsule.
        layer_type: string, one of 'FC' or "CONV", the type of this layer,
            fully connected or convolution, for the future expansion capability
        with_routing: boolean, this capsule is routing with the
                      lower-level layer capsule.

    Returns:
        A 4-D tensor.
    '''
    def __init__(self, num_outputs, vec_len, with_routing=True, layer_type='FC'):
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.with_routing = with_routing
        self.layer_type = layer_type

    def __call__(self, input, kernel_size=None, stride=None):
        '''
        The parameters 'kernel_size' and 'stride' will be used while 'layer_type' equal 'CONV'
        '''
        
        if self.layer_type == 'FC':
            if self.with_routing:
                # the DigitCaps layer, a fully connected layer
                # Reshape the input into [batch_size, 1152, 1, 8, 1]
                self.input = tf.reshape(input, shape=(batch_size, -1, 1, lc_out_channel, 1))

                with tf.variable_scope('routing'):
                    # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
                    # about the reason of using 'batch_size', see issue #21
                    b_IJ = tf.constant(np.zeros([batch_size, lc_out_h*lc_out_w, self.num_outputs, 1, 1], dtype=np.float32))
                    capsules = routing(self.input, b_IJ)
                    capsules = tf.squeeze(capsules, axis=1)

            return(capsules)
def cjl_net(x, weights, biases):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Local Convolution Layer
    lc1 = local_conv2d(x, weights['wc1'],
                       kernel_size=(lc_kernel_size, lc_kernel_size),
                       strides=(lc_stride, lc_stride),
                       output_shape=(lc_out_h, lc_out_w),
                       output_channel=lc_out_channel)
    lc1 = tf.nn.relu(lc1)   #[Batch, new_rows, new_cols,Channels)

    digitCaps = CapsLayer(num_outputs=10, vec_len=16, with_routing=True, layer_type='FC')
    caps2 = digitCaps(lc1)
    # Decoder structure in Fig. 2
    # 1. Do masking, how:
    with tf.variable_scope('Masking'):
        # a). calc ||v_c||, then do softmax(||v_c||)
        # [batch_size, 10, 16, 1] => [batch_size, 10, 1, 1]
        v_length = tf.sqrt(reduce_sum(tf.square(caps2),
                                               axis=2, keepdims=True) + epsilon)
        softmax_v = softmax(v_length, axis=1)
        assert softmax_v.get_shape() == [batch_size, 10, 1, 1]

            # b). pick out the index of max softmax val of the 10 caps
            # [batch_size, 10, 1, 1] => [batch_size] (index)
        argmax_idx = tf.to_int32(tf.argmax(softmax_v, axis=1))
        assert argmax_idx.get_shape() == [batch_size, 1, 1]
        argmax_idx = tf.reshape(argmax_idx, shape=(batch_size, ))

            # Method 1.
        if not cfg.mask_with_y:
                # c). indexing
                # It's not easy to understand the indexing process with argmax_idx
                # as we are 3-dim animal
            masked_v = []
            for batch_size_t in range(batch_size):
                v = caps2[batch_size_t][argmax_idx[batch_size_t], :]
                masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))

            masked_v = tf.concat(masked_v, axis=0)
            assert masked_v.get_shape() == [batch_size, 1, 16, 1]
        # Method 2. masking with true label, default mode
        else:
            # self.masked_v = tf.matmul(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 10, 1)), transpose_a=True)
            masked_v = tf.multiply(tf.squeeze(caps2), tf.reshape(Y, (-1, 10, 1)))
            v_length = tf.sqrt(reduce_sum(tf.square(caps2), axis=2, keepdims=True) + epsilon)

        # 2. Reconstructe the MNIST images with 3 FC layers
        # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
    with tf.variable_scope('Decoder'):
        vector_j = tf.reshape(masked_v, shape=(batch_size, -1))
        fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
        assert fc1.get_shape() == [batch_size, 512]
        fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
        assert fc2.get_shape() == [batch_size, 1024]
        decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)
    
    # # Fully connected layer
    # # Reshape lc1 output to fit fully connected layer input
    # fc1 = tf.reshape(lc1, [-1, weights['wd1'].get_shape().as_list()[0]])
    # fc1 = tf.matmul(fc1, weights['wd1'])
    # fc1 = tf.nn.relu(fc1)

    # # Output, class prediction
    # out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


pretrain_fn = "pretrainW_ksize_" + str(lc_kernel_size) + "_stride_" + str(lc_stride) + "_channel_" + str(lc_out_channel) + ".npy"
pretrain = np.load(pretrain_fn)

print(pretrain.shape)

# Store layers weight & bias
weights = {
    # 15x15 local conv, 1 input, 1 outputs
    'wc1': tf.Variable(tf.convert_to_tensor(pretrain, dtype=tf.float32)),
    'wd1': tf.Variable(tf.random_normal([lc_out_num * lc_out_channel, dense_size])),
    # 16 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([dense_size, num_classes]))
}

biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
logits = cjl_net(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    tf.stop_gradient(weights['wc1'])

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " +
                  "{:.4f}".format(loss) + ", Training Accuracy= " +
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={X: mnist.test.images,
                                        Y: mnist.test.labels}))
