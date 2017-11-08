import tensorflow


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tensorflow.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tensorflow.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tensorflow.truncated_normal(shape, stddev=0.1)
    return tensorflow.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tensorflow.constant(0.1, shape=shape)
    return tensorflow.Variable(initial)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tensorflow.name_scope('summaries'):
        mean = tensorflow.reduce_mean(var)
        tensorflow.summary.scalar('mean', mean)
        with tensorflow.name_scope('stddev'):
            stddev = tensorflow.sqrt(tensorflow.reduce_mean(tensorflow.square(var - mean)))
            tensorflow.summary.scalar('stddev', stddev)
        tensorflow.summary.scalar('max', tensorflow.reduce_max(var))
        tensorflow.summary.scalar('min', tensorflow.reduce_min(var))
        tensorflow.summary.histogram('histogram', var)


def get_scope_variable(name_scope, var, shape=None, initializer=None):
    with tensorflow.variable_scope(name_scope) as scope:
        try:
            v = tensorflow.get_variable(var, shape, initializer=initializer)
        except ValueError:
            scope.reuse_variables()
            v = tensorflow.get_variable(var)
    return v
