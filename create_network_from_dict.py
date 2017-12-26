import tensorflow as tf
import tensorboard


class network(object):
    def __init__(self, name, inputs, arch_dict):
        self.name = name
        self.inputs = inputs
        self.architecture = {}
        self.create_network(arch_dict)

    def conv_layer(self, inputs, channels_num, filter_amount, activation_function, filter_size=[5, 5], stride=1,
                   name="conv"):
        with tf.name_scope(name):
            w = tf.Variable(
                tf.truncated_normal([filter_size[0], filter_size[1], channels_num, filter_amount], stddev=0.1),
                name="W")
            b = tf.Variable(tf.ones([filter_amount]) / 10, name="B")
            tf.summary.histogram("W", w)
            tf.summary.histogram("B", b)
            return activation_function(
                tf.nn.conv2d(inputs, w, strides=[1, stride, stride, 1], padding="SAME") + b)

    def fully_connected(self, inputs, output_degree, activation_function, pkeep=1, name="fc"):
        with tf.name_scope(name):
            input_degree = inputs.shape[-1]
            w = tf.Variable(tf.truncated_normal([input_degree, output_degree], stddev=0.1), name="W")
            b = tf.Variable(tf.ones([output_degree]) / 10, name="B")
            tf.summary.histogram("W", w)
            tf.summary.histogram("B", b)
            act = tf.nn.dropout(activation_function(tf.matmul(inputs, w) + b), pkeep, name=name)
            tf.summary.histogram("activation function", act)
            return act

    def create_network(self, dict):
        for key,val in dict:
            if key.startswith("fc"):
                self.architecture[key] = self.fully_connected(inputs=val[0], output_degree=val[2], activation_function=val[3])
            elif key == "conv":
                return conv_layer(inputs=val[0], channels_num=val[1], filter_amount=val[3], activation_function=val[4],
                                  filter_size=val[5], stride=val[6])


