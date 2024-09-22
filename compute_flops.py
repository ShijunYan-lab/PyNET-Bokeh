import tensorflow as tf

from model import stack

# 禁用 eager execution 以兼容 TensorFlow 1.x 风格的代码
tf.compat.v1.disable_eager_execution()

# 用于累积 FLOPs 计算的全局变量
total_flops = 0

# 计算卷积层 FLOPs 的函数
def compute_conv_flops(input_shape, output_shape, kernel_size, in_channels, out_channels):
    """计算卷积层的 FLOPs"""
    flops = output_shape[1] * output_shape[2] * in_channels * out_channels * (kernel_size ** 2)
    return flops

# 计算转置卷积层 FLOPs 的函数
def compute_deconv_flops(input_shape, output_shape, kernel_size, in_channels, out_channels):
    """计算转置卷积层的 FLOPs"""
    flops = output_shape[1] * output_shape[2] * in_channels * out_channels * (kernel_size ** 2)
    return flops

def _conv_tranpose_layer(net, num_filters, filter_size, strides):
    """实现转置卷积（transpose convolution）"""
    input_shape = tf.shape(net)
    output_shape = [input_shape[0], input_shape[1] * strides, input_shape[2] * strides, num_filters]

    # 定义转置卷积核权重
    weights_init = tf.Variable(tf.compat.v1.truncated_normal(
        [filter_size, filter_size, num_filters, net.get_shape()[-1]], stddev=0.01), dtype=tf.float32)

    # 执行转置卷积
    net = tf.nn.conv2d_transpose(net, weights_init, output_shape, strides=[1, strides, strides, 1], padding='SAME')

    # 应用激活函数
    net = leaky_relu(net)
    return net

def PyNET(input, instance_norm=True, instance_norm_level_1=False):
    global total_flops  # 使用全局变量来跟踪 FLOPs

    with tf.compat.v1.variable_scope("generator"):
        # -----------------------------------------
        # Space-to-depth layer
        space2depth_l0 = tf.nn.space_to_depth(input, 2)  # 512 -> 256

        # Downsampling layers
        conv_l1_d1 = _conv_multi_block(space2depth_l0, 3, num_maps=32, instance_norm=False)  # 256 -> 256
        pool1 = max_pool(conv_l1_d1, 2)  # 256 -> 128

        input_shape = space2depth_l0.get_shape().as_list()
        output_shape = conv_l1_d1.get_shape().as_list()
        total_flops += compute_conv_flops(input_shape, output_shape, 3, input_shape[-1], 32)

        conv_l2_d1 = _conv_multi_block(pool1, 3, num_maps=64, instance_norm=instance_norm)  # 128 -> 128
        pool2 = max_pool(conv_l2_d1, 2)  # 128 -> 64

        input_shape = pool1.get_shape().as_list()
        output_shape = conv_l2_d1.get_shape().as_list()
        total_flops += compute_conv_flops(input_shape, output_shape, 3, input_shape[-1], 64)

        conv_l3_d1 = _conv_multi_block(pool2, 3, num_maps=128, instance_norm=instance_norm)  # 64 -> 64
        pool3 = max_pool(conv_l3_d1, 2)  # 64 -> 32

        input_shape = pool2.get_shape().as_list()
        output_shape = conv_l3_d1.get_shape().as_list()
        total_flops += compute_conv_flops(input_shape, output_shape, 3, input_shape[-1], 128)

        conv_l4_d1 = _conv_multi_block(pool3, 3, num_maps=256, instance_norm=instance_norm)  # 32 -> 32
        pool4 = max_pool(conv_l4_d1, 2)  # 32 -> 16

        input_shape = pool3.get_shape().as_list()
        output_shape = conv_l4_d1.get_shape().as_list()
        total_flops += compute_conv_flops(input_shape, output_shape, 3, input_shape[-1], 256)

        # Processing: Level 5, Input size: 16 x 16
        conv_l5_d1 = _conv_multi_block(pool4, 3, num_maps=512, instance_norm=instance_norm)
        input_shape = pool4.get_shape().as_list()
        output_shape = conv_l5_d1.get_shape().as_list()
        total_flops += compute_conv_flops(input_shape, output_shape, 3, input_shape[-1], 512)

        conv_l5_d2 = _conv_multi_block(conv_l5_d1, 3, num_maps=512, instance_norm=instance_norm)
        input_shape = conv_l5_d1.get_shape().as_list()
        output_shape = conv_l5_d2.get_shape().as_list()
        total_flops += compute_conv_flops(input_shape, output_shape, 3, input_shape[-1], 512)

        conv_l5_d3 = _conv_multi_block(conv_l5_d2, 3, num_maps=512, instance_norm=instance_norm)
        input_shape = conv_l5_d2.get_shape().as_list()
        output_shape = conv_l5_d3.get_shape().as_list()
        total_flops += compute_conv_flops(input_shape, output_shape, 3, input_shape[-1], 512)

        conv_l5_d4 = _conv_multi_block(conv_l5_d3, 3, num_maps=512, instance_norm=instance_norm)
        input_shape = conv_l5_d3.get_shape().as_list()
        output_shape = conv_l5_d4.get_shape().as_list()
        total_flops += compute_conv_flops(input_shape, output_shape, 3, input_shape[-1], 512)


        conv_l5_out = _conv_layer(conv_l5_d4, 3, 3, 1, relu=False, instance_norm=False)
        output_l5 = tf.nn.tanh(conv_l5_out) * 0.58 + 0.5
        total_flops += compute_conv_flops(conv_l5_d4.get_shape().as_list(), pool4.get_shape().as_list(), 3,
                                          conv_l5_d4.get_shape()[-1], 3)

        # Transposed Convolution (conv_t4a and conv_t4b), size: 16 -> 32
        conv_t4a = _conv_tranpose_layer(conv_l5_d4, 256, 3, 2)  # 16 -> 32
        conv_t4b = _conv_tranpose_layer(conv_l5_d4, 256, 3, 2)  # 16 -> 32

        input_shape = conv_l5_d1.get_shape().as_list()
        output_shape_t4a = conv_t4a.get_shape().as_list()
        output_shape_t4b = conv_t4b.get_shape().as_list()
        total_flops += compute_deconv_flops(input_shape, output_shape_t4a, 3, input_shape[-1], 256)
        total_flops += compute_deconv_flops(input_shape, output_shape_t4b, 3, input_shape[-1], 256)

        # conv_l4_x series layers
        conv_l4_d2 = stack(conv_l4_d1, conv_t4a)
        conv_l4_d3 = _conv_multi_block(conv_l4_d2, 3, num_maps=256, instance_norm=instance_norm)
        conv_l4_d4 = _conv_multi_block(conv_l4_d3, 3, num_maps=256, instance_norm=instance_norm) + conv_l4_d3
        conv_l4_d5 = _conv_multi_block(conv_l4_d4, 3, num_maps=256, instance_norm=instance_norm) + conv_l4_d4
        conv_l4_d6 = stack(_conv_multi_block(conv_l4_d5, 3, num_maps=256, instance_norm=instance_norm), conv_t4b)
        conv_l4_d7 = _conv_multi_block(conv_l4_d6, 3, num_maps=256, instance_norm=instance_norm)

        # 计算 conv_l4 层的 FLOPs
        input_shape = conv_l4_d1.get_shape().as_list()
        output_shape = conv_l4_d7.get_shape().as_list()
        total_flops += compute_conv_flops(input_shape, output_shape, 3, input_shape[-1], 256)

        # -> Output: Level 4
        conv_l4_out = _conv_layer(conv_l4_d7, 3, 3, 1, relu=False, instance_norm=False)
        output_l4 = tf.nn.tanh(conv_l4_out) * 0.58 + 0.5
        total_flops += compute_conv_flops(conv_l4_d7.get_shape().as_list(), conv_l4_out.get_shape().as_list(), 3, conv_l4_d7.get_shape()[-1], 3)

        # 继续添加其他层的计算量...
        conv_t3a = _conv_tranpose_layer(conv_l4_d7, 128, 3, 2)  # 32 -> 64
        conv_t3b = _conv_tranpose_layer(conv_l4_d7, 128, 3, 2)  # 32 -> 64


        # conv_l3_x series layers
        conv_l3_d2 = stack(conv_l3_d1, conv_t3a)
        conv_l3_d3 = _conv_multi_block(conv_l3_d2, 5, num_maps=128, instance_norm=instance_norm)
        conv_l3_d4 = _conv_multi_block(conv_l3_d3, 5, num_maps=128, instance_norm=instance_norm) + conv_l3_d3
        conv_l3_d5 = _conv_multi_block(conv_l3_d4, 5, num_maps=128, instance_norm=instance_norm) + conv_l3_d4
        conv_l3_d6 = stack(_conv_multi_block(conv_l3_d5, 5, num_maps=128, instance_norm=instance_norm), conv_l3_d1)
        conv_l3_d7 = stack(conv_l3_d6, conv_t3b)
        conv_l3_d8 = _conv_multi_block(conv_l3_d7, 3, num_maps=128, instance_norm=instance_norm)

        # 计算 conv_l3 层的 FLOPs
        input_shape = conv_l3_d1.get_shape().as_list()
        output_shape = conv_l3_d8.get_shape().as_list()
        total_flops += compute_conv_flops(input_shape, output_shape, 3, input_shape[-1], 256)

        # -> Output: Level 3
        conv_l3_out = _conv_layer(conv_l3_d8, 3, 3, 1, relu=False, instance_norm=False)
        output_l3 = tf.nn.tanh(conv_l3_out) * 0.58 + 0.5
        total_flops += compute_conv_flops(conv_l3_d8.get_shape().as_list(), conv_l3_out.get_shape().as_list(), 3, conv_l3_d8.get_shape()[-1], 3)

        conv_t2a = _conv_tranpose_layer(conv_l3_d8, 64, 3, 2)  # 64 -> 128
        conv_t2b = _conv_tranpose_layer(conv_l3_d8, 64, 3, 2)  # 64 -> 128
        conv_l2_d2 = stack(conv_l2_d1, conv_t2a)
        conv_l2_d3 = stack(_conv_multi_block(conv_l2_d2, 5, num_maps=64, instance_norm=instance_norm), conv_l2_d1)

        conv_l2_d4 = _conv_multi_block(conv_l2_d3, 7, num_maps=64, instance_norm=instance_norm)
        conv_l2_d5 = _conv_multi_block(conv_l2_d4, 7, num_maps=64, instance_norm=instance_norm) + conv_l2_d4
        conv_l2_d6 = _conv_multi_block(conv_l2_d5, 7, num_maps=64, instance_norm=instance_norm) + conv_l2_d5
        conv_l2_d7 = stack(_conv_multi_block(conv_l2_d6, 7, num_maps=64, instance_norm=instance_norm), conv_l2_d1)

        conv_l2_d8 = stack(_conv_multi_block(conv_l2_d7, 5, num_maps=64, instance_norm=instance_norm), conv_t2b)
        conv_l2_d9 = _conv_multi_block(conv_l2_d8, 3, num_maps=64, instance_norm=instance_norm)

        # 计算 conv_l2 层的 FLOPs
        input_shape = conv_l2_d1.get_shape().as_list()
        output_shape = conv_l2_d9.get_shape().as_list()
        total_flops += compute_conv_flops(input_shape, output_shape, 3, input_shape[-1], 256)

        # -> Output: Level 2
        conv_l2_out = _conv_layer(conv_l2_d9, 3, 3, 1, relu=False, instance_norm=False)
        output_l2 = tf.nn.tanh(conv_l2_out) * 0.58 + 0.5
        total_flops += compute_conv_flops(conv_l2_d9.get_shape().as_list(), conv_l2_out.get_shape().as_list(), 3,
                                          conv_l2_d9.get_shape()[-1], 3)

        conv_t1a = _conv_tranpose_layer(conv_l2_d9, 32, 3, 2)  # 128 -> 256
        conv_t1b = _conv_tranpose_layer(conv_l2_d9, 32, 3, 2)  # 128 -> 256

        conv_l1_d2 = stack(conv_l1_d1, conv_t1a)
        conv_l1_d3 = stack(_conv_multi_block(conv_l1_d2, 5, num_maps=32, instance_norm=False), conv_l1_d1)
        conv_l1_d4 = _conv_multi_block(conv_l1_d3, 7, num_maps=32, instance_norm=False)
        conv_l1_d5 = _conv_multi_block(conv_l1_d4, 9, num_maps=32, instance_norm=instance_norm_level_1)
        conv_l1_d6 = _conv_multi_block(conv_l1_d5, 9, num_maps=32, instance_norm=instance_norm_level_1) + conv_l1_d5
        conv_l1_d7 = _conv_multi_block(conv_l1_d6, 9, num_maps=32, instance_norm=instance_norm_level_1) + conv_l1_d6
        conv_l1_d8 = _conv_multi_block(conv_l1_d7, 9, num_maps=32, instance_norm=instance_norm_level_1) + conv_l1_d7
        conv_l1_d9 = stack(_conv_multi_block(conv_l1_d8, 7, num_maps=32, instance_norm=False), conv_l1_d1)
        conv_l1_d10 = stack(_conv_multi_block(conv_l1_d9, 5, num_maps=32, instance_norm=False), conv_t1b)
        conv_l1_d11 = stack(conv_l1_d10, conv_l1_d1)
        conv_l1_d12 = _conv_multi_block(conv_l1_d11, 3, num_maps=32, instance_norm=False)

        # 计算 conv_l1 层的 FLOPs
        input_shape = conv_l1_d1.get_shape().as_list()
        output_shape = conv_l1_d12.get_shape().as_list()
        total_flops += compute_conv_flops(input_shape, output_shape, 3, input_shape[-1], 256)

        # -> Output: Level 1
        conv_l1_out = _conv_layer(conv_l1_d12, 3, 3, 1, relu=False, instance_norm=False)
        output_l1 = tf.nn.tanh(conv_l1_out) * 0.58 + 0.5
        total_flops += compute_conv_flops(conv_l1_d12.get_shape().as_list(), conv_l1_out.get_shape().as_list(), 3,
                                          conv_l1_d12.get_shape()[-1], 3)

        conv_l0 = _conv_tranpose_layer(conv_l1_d12, 8, 3, 2)  # 256 -> 512
        conv_l0_out = _conv_layer(conv_l0, 3, 3, 1, relu=False, instance_norm=False)
        output_l0 = tf.nn.tanh(conv_l0_out) * 0.58 + 0.5

        # -> Output: Level Up (final output layers)
        conv_l_up = _conv_tranpose_layer(conv_l0_out, 3, 3, 2)  # 512 -> 1024
        conv_l_up_out = _conv_layer(conv_l_up, 3, 3, 1, relu=False, instance_norm=False)
        output_l_up = tf.nn.tanh(conv_l_up_out) * 0.58 + 0.5

        # 计算最终输出层的 FLOPs
        total_flops += compute_deconv_flops(conv_l0_out.get_shape().as_list(), conv_l_up.get_shape().as_list(), 3, conv_l0_out.get_shape()[-1], 3)
        total_flops += compute_conv_flops(conv_l_up.get_shape().as_list(), conv_l_up_out.get_shape().as_list(), 3, conv_l_up.get_shape()[-1], 3)

    return output_l_up, output_l0, output_l1, output_l2, output_l3, output_l4, output_l5

# 定义 max_pool 函数
def max_pool(x, ksize):
    return tf.nn.max_pool2d(x, ksize=ksize, strides=ksize, padding='SAME')

# Conv multi block
def _conv_multi_block(input, max_size, num_maps, instance_norm):
    conv_3a = _conv_layer(input, num_maps, 3, 1, relu=True, instance_norm=instance_norm)
    return conv_3a

# Conv layer
def _conv_layer(net, num_filters, filter_size, strides, relu=True, instance_norm=False, padding='SAME'):
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    bias = tf.Variable(tf.constant(0.01, shape=[num_filters]))

    net = tf.nn.conv2d(net, weights_init, strides=strides_shape, padding=padding) + bias

    if instance_norm:
        net = _instance_norm(net)

    if relu:
        net = leaky_relu(net)

    return net

# Conv init vars
def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    in_channels = net.get_shape()[-1]  # 静态形状
    weights_shape = [filter_size, filter_size, in_channels, out_channels]

    weights_init = tf.Variable(tf.compat.v1.truncated_normal(weights_shape, stddev=0.01, seed=1), dtype=tf.float32)
    return weights_init

# Instance norm
def _instance_norm(net):
    batch, rows, cols, channels = [i for i in net.get_shape()]
    var_shape = [channels]

    mu, sigma_sq = tf.compat.v1.nn.moments(net, [1, 2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net - mu) / (sigma_sq + epsilon) ** 0.5
    return scale * normalized + shift

# Leaky ReLU
def leaky_relu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)

# 计算 FLOPs 的函数
def get_flops(model_func, input_shape):
    # Reset graph
    tf.compat.v1.reset_default_graph()

    # Define input placeholder
    input_tensor = tf.compat.v1.placeholder(tf.float32, shape=input_shape, name='input_image')

    # Build the model
    output = model_func(input_tensor)

    # 打印总 FLOPs
    print(f"Total FLOPs: {total_flops}")

# 定义输入的形状
input_shape = [1, 512, 512, 3]  # Batch size = 1, Image size = 512x512, 3 channels (RGB)

# 获取 FLOPs 计算量
get_flops(PyNET, input_shape)
