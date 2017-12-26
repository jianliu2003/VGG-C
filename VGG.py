import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

PATH = "/home/student-9/PycharmProjects/Axon-Course/data_dir/miny_set/tiny-imagenet-200"


class VGG(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 64, 64, 3])
        self.output_size = 100
        self.Y_ = tf.placeholder(tf.float32, [None, self.output_size])
        self.lr = tf.placeholder(tf.float32)
        self.max_step = 2000000
        self.sess = tf.Session()
        self.batch_size = 256
        self.regularizer = 0
        self.multiplier = 0.0005
        self.pkeep = 0.75
        self.sess_name = 'VGG_tb_pk075_xavier_c100'
        self.fc_size = 4096

        # layer 1
        conv1_1 = self.conv_layer(self.x, 3, 64, tf.nn.relu, filter_size=[3, 3], stride=1, name="conv1_1")
        conv1_2 = self.conv_layer(conv1_1, 64, 64, tf.nn.relu, filter_size=[3, 3], stride=1, name="conv1_2")
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")
        # layer 2
        conv2_1 = self.conv_layer(pool1, 64, 128, tf.nn.relu, filter_size=[3, 3], stride=1, name="conv2_1")
        conv2_2 = self.conv_layer(conv2_1, 128, 128, tf.nn.relu, filter_size=[3, 3], stride=1, name="conv2_2")
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")
        # layer 3
        conv3_1 = self.conv_layer(pool2, 128, 256, tf.nn.relu, filter_size=[3, 3], stride=1, name="conv3_1")
        conv3_2 = self.conv_layer(conv3_1, 256, 256, tf.nn.relu, filter_size=[3, 3], stride=1, name="conv3_2")
        conv3_3 = self.conv_layer(conv3_2, 256, 256, tf.nn.relu, filter_size=[1, 1], stride=1, name="conv3_3")
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool3")
        # layer 4
        conv4_1 = self.conv_layer(pool3, 256, 512, tf.nn.relu, filter_size=[3, 3], stride=1, name="conv4_1")
        conv4_2 = self.conv_layer(conv4_1, 512, 512, tf.nn.relu, filter_size=[3, 3], stride=1, name="conv4_2")
        conv4_3 = self.conv_layer(conv4_2, 512, 512, tf.nn.relu, filter_size=[1, 1], stride=1, name="conv4_3")
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool4")
        # layer 5
        conv5_1 = self.conv_layer(pool4, 512, 512, tf.nn.relu, filter_size=[3, 3], stride=1, name="conv5_1")
        conv5_2 = self.conv_layer(conv5_1, 512, 512, tf.nn.relu, filter_size=[3, 3], stride=1, name="conv5_2")
        conv5_3 = self.conv_layer(conv5_2, 512, 512, tf.nn.relu, filter_size=[1, 1], stride=1, name="conv5_3")
        pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool5")
        # layer FC
        fc6_input = tf.reshape(pool5, [-1, pool5.shape[1] * pool5.shape[2] * pool5.shape[3]])  # 2048
        # print fc4_input.shape[1]
        fc6 = self.fully_connected(fc6_input, int(fc6_input.shape[1]), self.fc_size, tf.nn.relu, pkeep=self.pkeep, name="fc6")
        fc7 = self.fully_connected(fc6, self.fc_size, self.fc_size, tf.nn.relu, pkeep=self.pkeep, name="fc7") ##
        logits = self.fully_connected(fc7, self.fc_size, self.output_size, tf.identity, name="fc8")
        self.y = tf.nn.softmax(logits)
        with tf.name_scope("loss"):
            with tf.name_scope("cross-entropy"):
                self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.Y_))
            self.loss = self.cross_entropy +  self.multiplier*self.regularizer
        with tf.name_scope("accuracy"):
            is_correct = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.Y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        tf.summary.scalar("accuracy", self.accuracy)
        tf.summary.scalar("cross_entropy", self.cross_entropy)
        tf.summary.scalar("loss", self.loss)
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.saver = tf.train.Saver()

        init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()
        self.merged = tf.summary.merge_all()
        self.train_images, self.train_cls_vec, self.test_images, self.test_cls_vec = self.prepare_imagenet_data()
        self.sess = tf.Session()
        self.sess.run(init)
        self.writer_train = tf.summary.FileWriter('./tensorboard/%s/train' %self.sess_name, self.sess.graph)
        self.writer_test = tf.summary.FileWriter('./tensorboard/%s/test' %self.sess_name)
        # self.writer.add_graph(self.sess.graph)

    def prepare_imagenet_data(self):
        from load_images import load_images
        from sklearn import preprocessing
        num_classes = self.output_size  # max number
        X_train, y_train, X_test, y_test = load_images(PATH, num_classes)
        lb = preprocessing.LabelBinarizer()
        y_train_onehot = lb.fit_transform(y_train)
        y_test_onehot = lb.fit_transform(y_test)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        # print type(X_test), X_test.shape
        # plt.imshow(X_train[1,:, :, :])
        # plt.show()
        # import pdb
        # pdb.set_trace()
        return X_train, y_train_onehot, X_test, y_test_onehot

    def get_batch_data(self, x, y, batch_size):
        batch_idxs = np.random.choice(range(x.shape[0]), batch_size, replace=False)
        batch_x = x[batch_idxs, :, :, :]
        batch_y = y[batch_idxs, :]
        return batch_x, batch_y

    def train_net(self):
        lr = 0.001
        for i in range(self.max_step):
            with tf.name_scope("train"):
                X_data, Y_data = self.get_batch_data(self.train_images, self.train_cls_vec, self.batch_size)
                if i % 700==0: # 0.01 -> 0.00001
                    lr = lr / 10.
                    save_path = self.saver.save(self.sess, "./tensorboard/saved_model/%s_%d.ckpt" %(self.sess_name, i))
                train_dict = {self.x: X_data, self.Y_: Y_data, self.lr: lr}
                self.sess.run(self.train_step, feed_dict=train_dict)
            if i % 10 == 0:
                train_summ = self.test(X_data, Y_data)
                self.writer_train.add_summary(train_summ, i)
                print i
            if i % 10 == 0:
                X_test, Y_test = self.get_batch_data(self.test_images, self.test_cls_vec, self.batch_size)
                # test_summ = self.test(self.test_images, self.test_cls_vec) #X_test, Y_test)
                test_summ = self.test(X_test, Y_test)
                self.writer_test.add_summary(test_summ, i)

    def test(self, X_data, Y_data):
        with tf.name_scope("test"):
            # X_data, Y_data = self.test_images, self.test_cls_vec
            test_dict = {self.x: X_data, self.Y_: Y_data}
            tf.summary.image("test_images", X_data)
            return self.sess.run(self.merged, feed_dict=test_dict)


    def conv_layer(self, inputs, channels_num, filter_amount, activation_function, filter_size=[5, 5], stride=1,
                   name="conv"):
        with tf.name_scope(name):
            # w = tf.Variable(
            #     tf.truncated_normal([filter_size[0], filter_size[1], channels_num, filter_amount], stddev=0.01),
            #     name="W")
            initializer = tf.contrib.layers.xavier_initializer_conv2d()
            w = tf.Variable(
                initializer(shape=[filter_size[0], filter_size[1], channels_num, filter_amount]),
                name="W")
            b = tf.Variable(tf.ones([filter_amount]) / 100, name="B")
            tf.summary.histogram("W", w)
            tf.summary.histogram("B", b)
            self.regularizer += tf.nn.l2_loss(w)
            return activation_function(
                tf.nn.conv2d(inputs, w, strides=[1, stride, stride, 1], padding="SAME") + b)

    def fully_connected(self, inputs, input_degree, output_degree, activation_function, pkeep=1, name="fc"):
        with tf.name_scope(name):
            initializer = tf.contrib.layers.xavier_initializer()
            #w = tf.Variable(tf.truncated_normal([input_degree, output_degree], stddev=0.01), name="W")
            w = tf.Variable(initializer(shape=[input_degree, output_degree]), name="W")
            b = tf.Variable(tf.ones([output_degree]) / 100, name="B")
            tf.summary.histogram("W", w)
            tf.summary.histogram("B", b)
            self.regularizer += tf.nn.l2_loss(w)
            act = tf.nn.dropout(activation_function(tf.matmul(inputs, w) + b), pkeep, name=name)
            tf.summary.histogram("activation function", act)
            return act


def main():
    VGG_net = VGG()
    VGG_net.train_net()


if __name__ == "__main__":
    main()


# tensorboard --logdir=/home/student-9/PycharmProjects/Axon-Course/tensorboard/VGG_tb