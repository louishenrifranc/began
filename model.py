import tensorflow as tf
import tensorflow.contrib.layers as ly
import helper
import tf_utils
from tqdm import trange
import numpy as np


class Graph:
    def __init__(self, args):
        self.batch_size = args.train.batch_size
        self.nb_epochs = args.train.nb_epochs

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # Placeholders
        self.is_training = tf.placeholder_with_default(False, shape=(), name="is_training")
        self.learning_rate = tf.placeholder_with_default(1e-5, shape=(), name="learning_rate")
        self.k_t = tf.placeholder_with_default(0.0, shape=[], name="k_t")
        self.carry = tf.train.exponential_decay(1.0, self.global_step,
                                                1000, 0.97, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Input filename
        self.cfg = args
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

    def _inputs(self):
        input_to_graph = helper.create_queue(self.cfg.queue.filename, self.batch_size)
        # True image
        self.true_image = input_to_graph
        # self.wrong_image = input_to_graph[6]
        # Sum of the caption
        # self.mean_caption = None
        # for i in range(1, 6):
        #     input_to_graph[i] = tf.transpose(input_to_graph[i], [0, 2, 1])
        #     self.mean_caption = input_to_graph[i] if self.mean_caption is None else \
        #         tf.concat([self.mean_caption, input_to_graph[i]], axis=1)
        #
        # self.sum_caption = tf.reduce_sum(self.mean_caption, axis=1)

    def build(self):
        self._inputs()

        # self._mean, self._log_sigma = self._generate_condition(self.sum_caption)

        # Sample conditioning from a Gaussian distribution parametrized by a Neural Network
        # z = helper.sample(self._mean, self._log_sigma)
        z = tf.random_normal((self.batch_size, self.cfg.emb.emb_dim), 0, 1)

        # z = tf.concat([z, self.emb], 1)
        # Generate an image
        self.gen_images = self.generator(z)

        # Decode the image
        self.reconstructed_true_image = self.discriminator(self.true_image)
        # self.reconstructed_wrong_image = self.discriminator(self.wrong_image, reuse_variables=True)
        self.reconstructed_gen_image = self.discriminator(self.gen_images, reuse_variables=True)

        self.adversarial_loss()
        self._summaries()

    def _generate_condition(self, sentence_embedding, scope_name="generate_condition", scope_reuse=False):
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()
            out = ly.fully_connected(sentence_embedding,
                                     self.cfg.emb.emb_dim * 2,
                                     activation_fn=tf_utils.leaky_rectify)
            return out

    def generator(self, z, scope_name="generator", reuse_variables=False):
        with tf.variable_scope(scope_name) as scope:
            if reuse_variables:
                scope.reuse_variables()
            n = self.cfg.emb.emb_dim
            out = ly.fully_connected(z, 8 * 8 * n, activation_fn=tf_utils.leaky_rectify,
                                     normalizer_fn=ly.batch_norm)

            in_x = tf.reshape(out, [-1, 8, 8, n])

            out = tf_utils.cust_conv2d(in_x, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out1")
            out = self.carry * in_x + (1 - self.carry) * out
            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out2")

            in_x = tf.image.resize_nearest_neighbor(out, [16, 16])

            out = tf_utils.cust_conv2d(in_x, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out4")
            out = self.carry * in_x + (1 - self.carry) * out
            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out5")

            in_x = tf.image.resize_nearest_neighbor(out, [32, 32])

            out = tf_utils.cust_conv2d(in_x, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out6")
            out = self.carry * in_x + (1 - self.carry) * out
            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out7")

            in_x = tf.image.resize_nearest_neighbor(out, [64, 64])

            out = tf_utils.cust_conv2d(in_x, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out8")
            out = self.carry * in_x + (1 - self.carry) * out
            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out9")

            in_x = tf.image.resize_nearest_neighbor(out, [128, 128])

            out = tf_utils.cust_conv2d(in_x, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out10")
            out = self.carry * in_x + (1 - self.carry) * out
            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out11")

            out = tf_utils.cust_conv2d(out, 3, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=False, scope_name="out12",
                                       activation_fn=tf.tanh)
            return out

    def discriminator(self, img, scope_name="discriminator", reuse_variables=False):
        with tf.variable_scope(scope_name) as scope:
            if reuse_variables:
                scope.reuse_variables()

            n = self.cfg.emb.emb_dim

            ############################################# Encoder part #############################################
            # 128 x 128
            out = tf_utils.cust_conv2d(img, 3, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out1")
            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out2")

            # 64 x 64
            in_x = tf_utils.cust_conv2d(out, 2 * n, h_f=3, w_f=3, batch_norm=True, scope_name="down_1")
            out = tf_utils.cust_conv2d(in_x, 2 * n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out3")
            out = self.carry * in_x + (1 - self.carry) * out
            out = tf_utils.cust_conv2d(out, 2 * n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out4")

            # 32 x 32
            in_x = tf_utils.cust_conv2d(out, 3 * n, h_f=3, w_f=3, batch_norm=True, scope_name="down_1bis")
            out = tf_utils.cust_conv2d(in_x, 3 * n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out3bis")
            out = self.carry * in_x + (1 - self.carry) * out
            out = tf_utils.cust_conv2d(out, 3 * n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out4bis")

            # 16 x 16
            in_x = tf_utils.cust_conv2d(out, 4 * n, h_f=3, w_f=3, batch_norm=True, scope_name="down_2")
            out = tf_utils.cust_conv2d(in_x, 4 * n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out5")
            out = self.carry * in_x + (1 - self.carry) * out
            out = tf_utils.cust_conv2d(out, 4 * n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out6")

            # 8 x 8
            in_x = tf_utils.cust_conv2d(out, 5 * n, h_f=3, w_f=3, batch_norm=True, scope_name="down_3")
            out = tf_utils.cust_conv2d(in_x, 5 * n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out7")
            out = self.carry * in_x + (1 - self.carry) * out
            out = tf_utils.cust_conv2d(out, 5 * n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out8")

            # Concat embeddings
            out = tf.reshape(out, [-1, 8 * 8 * 5 * n])

            out = ly.fully_connected(out, 2000, activation_fn=tf_utils.leaky_rectify, normalizer_fn=ly.batch_norm)

            ############################################# Decoder part #############################################

            out = ly.fully_connected(out, 8 * 8 * n, activation_fn=tf_utils.leaky_rectify, normalizer_fn=ly.batch_norm)
            in_x = tf.reshape(out, [-1, 8, 8, n])

            out = tf_utils.cust_conv2d(in_x, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out9")
            out = self.carry * in_x + (1 - self.carry) * out
            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out10")

            in_x = tf.image.resize_nearest_neighbor(out, [16, 16])
            out = tf_utils.cust_conv2d(in_x, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out11")
            out = self.carry * in_x + (1 - self.carry) * out
            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out12")

            in_x = tf.image.resize_nearest_neighbor(out, [32, 32])
            out = tf_utils.cust_conv2d(in_x, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out13")
            out = self.carry * in_x + (1 - self.carry) * out
            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out14")

            in_x = tf.image.resize_nearest_neighbor(out, [64, 64])
            out = tf_utils.cust_conv2d(in_x, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out15")
            out = self.carry * in_x + (1 - self.carry) * out
            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=False, scope_name="out16")

            in_x = tf.image.resize_nearest_neighbor(out, [128, 128])
            out = tf_utils.cust_conv2d(in_x, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=False, scope_name="out16bis")
            out = self.carry * in_x + (1 - self.carry) * out
            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=False, scope_name="out17")

            out = tf_utils.cust_conv2d(out, 3, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=False, scope_name="out18",
                                       activation_fn=tf.tanh)
            return out

    def compute_loss(self, D_real_in, D_real_out, D_gen_in, D_gen_out, k_t, gamma=0.75):
        def autoencoder_loss(out, inp, eta=2):
            diff = tf.abs(out - inp)
            return tf.reduce_sum(tf.pow(diff, eta))

        mu_real = autoencoder_loss(D_real_out, D_real_in)
        mu_gen = autoencoder_loss(D_gen_out, D_gen_in)

        D_loss = mu_real - k_t * (1 / 2 * mu_gen)
        G_loss = (1 / 2 * mu_gen)

        lam = 0.001  # 'learning rate' for k. Berthelot et al. use 0.001
        k_tp = k_t + lam * (gamma * mu_real - mu_gen)
        convergence_measure = mu_real + np.abs(gamma * mu_real - mu_gen)
        return D_loss, G_loss, k_tp, convergence_measure

    def adversarial_loss(self, scope_name="losses"):
        with tf.variable_scope(scope_name) as scope:
            d_loss, g_loss, k_tp, convergence_measure = self.compute_loss(self.true_image,
                                                                          self.reconstructed_true_image,
                                                                          self.gen_images,
                                                                          self.reconstructed_gen_image,
                                                                          self.k_t)

            train_vars = tf.trainable_variables()

            gen_variables = [v for v in train_vars if not v.name.startswith("discriminator")]
            dis_variables = [v for v in train_vars if v.name.startswith("discriminator")]

            g_grad = self.optimizer.compute_gradients(g_loss, var_list=gen_variables)
            d_grad = self.optimizer.compute_gradients(d_loss, var_list=dis_variables)

            # self.kl_loss = -self._log_sigma + 0.5 * (-1 + tf.exp(2 * self._log_sigma) + tf.square(self._mean))
            # self.kl_loss = tf.reduce_mean(self.kl_loss)
            # g_loss += self.kl_loss

            # Training function
            self.g_train = self.optimizer.apply_gradients(g_grad, global_step=self.global_step)
            self.d_train = self.optimizer.apply_gradients(d_grad, global_step=self.global_step)

            # Loss function
            self.d_loss = d_loss
            self.g_loss = g_loss
            self.k_tp = k_tp

    def _summaries(self):
        """
        Helper to add summaries
        :return:
        """

        # Add summaries for images
        num_images = self.batch_size
        tf.summary.image(name="true_image", tensor=self.true_image, max_outputs=num_images)
        tf.summary.image(name="gen_image", tensor=self.gen_images, max_outputs=num_images)

        tf.summary.image(name="reconstructed_true_image", tensor=self.reconstructed_true_image, max_outputs=num_images)
        tf.summary.image(name="reconstructed_gen_image", tensor=self.reconstructed_gen_image, max_outputs=num_images)

        # Add summaries for loss functions
        tf.summary.scalar(name="g_loss", tensor=self.g_loss)
        tf.summary.scalar(name="d_loss", tensor=self.d_loss)
        tf.summary.scalar(name="k_tp", tensor=self.k_tp)
        # tf.summary.scalar(name="k_tp", tensor=self.k_tp)

        # Add summaries for exponential decaying variables
        tf.summary.scalar(name="learning_rate", tensor=self.learning_rate)

        self.merged_summary_op = tf.summary.merge_all()

    def train(self):
        self.saver, self.summary_writer = helper.restore(self)

        with self.sess.as_default() as sess:

            tf.train.start_queue_runners(sess=self.sess)
            coord = tf.train.Coordinator()

            nb_epochs = self.nb_epochs
            save_every = 3
            summary_every = 50
            nb_train_iter = 1000
            decay_every = 20
            k_t_ = 0

            last_epoch = self.global_step.eval() // nb_train_iter

            for epoch in trange(nb_epochs, desc="Epoch"):
                if epoch < last_epoch:
                    continue

                if coord.should_stop():
                    break

                for iter in trange(nb_train_iter):
                    learning_rate_ = 1e-5 * pow(0.5, epoch // decay_every)

                    input_feed = {
                        self.learning_rate: learning_rate_,
                        self.k_t: min(max(k_t_, 0), 1),
                        self.is_training: True
                    }
                    output_feed = [self.g_train, self.d_train, self.d_loss, self.g_loss, self.k_tp]

                    _, _, D_loss_, G_loss_, k_t_ = sess.run(output_feed, input_feed)

                    if iter % summary_every == 0:
                        summaries = self.merged_summary_op.eval()
                        self.summary_writer.add_summary(summaries, self.global_step.eval())

                if epoch % save_every == 0:
                    self.saver.save(sess, "model/", global_step=self.global_step.eval())

            coord.request_stop()
            coord.join()
