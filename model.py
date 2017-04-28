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
        self.learning_rate = tf.placeholder_with_default(0.00001, shape=(), name="learning_rate")
        self.k_t = tf.placeholder_with_default(0.0, shape=[], name="k_t")

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Input filename
        self.cfg = args
        self.sess = tf.Session()

    def _inputs(self):
        input_to_graph = helper.create_queue(self.cfg.queue.filename, self.batch_size)
        # True image
        self.true_image = input_to_graph[0]
        # Sum of the caption
        self.mean_caption = None
        for i in range(1, 6):
            input_to_graph[i] = tf.transpose(input_to_graph[i], [0, 2, 1])
            self.mean_caption = input_to_graph[i] if self.mean_caption is None else \
                tf.concat([self.mean_caption, input_to_graph[i]], axis=1)

        self.sum_caption = tf.reduce_sum(self.mean_caption, axis=1)

    def build(self):
        self._inputs()

        self._mean, self._log_sigma = self._generate_condition(self.sum_caption)

        # Sample conditioning from a Gaussian distribution parametrized by a Neural Network
        z = helper.sample(self._mean, self._log_sigma)

        # Generate an image
        self.gen_images = self.generator(z)

        # Decode the image
        self.reconstructed_true_image = self.discriminator(self.true_image, z)
        self.reconstructed_gen_image = self.discriminator(self.gen_images, z, reuse_variables=True)

        self.adversarial_loss()
        self._summaries()

    def _generate_condition(self, sentence_embedding, scope_name="generate_condition", scope_reuse=False):
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()
            out = ly.fully_connected(sentence_embedding,
                                     self.cfg.emb.emb_dim * 2,
                                     activation_fn=tf_utils.leaky_rectify)
            mean = out[:, :self.cfg.emb.emb_dim]
            log_sigma = out[:, self.cfg.emb.emb_dim:]
            # emb_dim
            return mean, log_sigma

    def generator(self, embedding, scope_name="generator", reuse_variables=False):
        with tf.variable_scope(scope_name) as scope:
            if reuse_variables:
                scope.reuse_variables()

            n = self.cfg.emb.emb_dim
            out = ly.fully_connected(embedding, 8 * 8 * n, activation_fn=tf_utils.leaky_rectify,
                                     normalizer_fn=ly.batch_norm)

            out = tf.reshape(out, [-1, 8, 8, n])

            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out1")
            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out2")
            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out3")

            out = tf.image.resize_nearest_neighbor(out, [16, 16])

            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out4")
            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out5")

            out = tf.image.resize_nearest_neighbor(out, [32, 32])

            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out6")
            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out7")

            out = tf.image.resize_nearest_neighbor(out, [64, 64])

            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out8")
            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out9")
            out = tf_utils.cust_conv2d(out, 3, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=False, scope_name="out10",
                                       activation_fn=tf.tanh)
            return out

    def discriminator(self, img, emb, scope_name="discriminator", reuse_variables=False):
        with tf.variable_scope(scope_name) as scope:
            if reuse_variables:
                scope.reuse_variables()

            n = self.cfg.emb.emb_dim

            ############################################# Encoder part #############################################
            # 64 x 64
            out = tf_utils.cust_conv2d(img, 3, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=False, scope_name="out1")
            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=False, scope_name="out2")

            # 32 x 32
            out = tf_utils.cust_conv2d(out, 2 * n, h_f=3, w_f=3, batch_norm=False, scope_name="down_1")
            out = tf_utils.cust_conv2d(out, 2 * n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=False, scope_name="out3")
            out = tf_utils.cust_conv2d(out, 2 * n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=False, scope_name="out4")

            # 16 x 16
            out = tf_utils.cust_conv2d(out, 3 * n, h_f=3, w_f=3, batch_norm=False, scope_name="down_2")
            out = tf_utils.cust_conv2d(out, 3 * n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=False, scope_name="out5")
            out = tf_utils.cust_conv2d(out, 3 * n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=False, scope_name="out6")

            # 8 x 8
            out = tf_utils.cust_conv2d(out, 4 * n, h_f=3, w_f=3, batch_norm=False, scope_name="down_3")
            out = tf_utils.cust_conv2d(out, 4 * n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=False, scope_name="out7")
            out = tf_utils.cust_conv2d(out, 4 * n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=False, scope_name="out8")

            # Concat embeddings
            emb = tf.expand_dims(tf.expand_dims(emb, 1), 1)
            emb = tf.tile(emb, [1, 8, 8, 1])
            out = tf.concat([out, emb], axis=3)
            out = tf.reshape(out, [-1, 8 * 8 * 5 * n])

            out = ly.fully_connected(out, 1900, activation_fn=tf_utils.leaky_rectify)

            ############################################# Decoder part #############################################
            out = ly.fully_connected(out, 8 * 8 * n // 2, activation_fn=tf_utils.leaky_rectify)
            out = tf.reshape(out, [-1, 8, 8, n // 2])

            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out9")
            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out10")

            out = tf.image.resize_nearest_neighbor(out, [16, 16])
            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out11")
            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out12")

            out = tf.image.resize_nearest_neighbor(out, [32, 32])
            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out13")
            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out14")

            out = tf.image.resize_nearest_neighbor(out, [64, 64])
            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out15")
            out = tf_utils.cust_conv2d(out, n, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=True, scope_name="out16")

            out = tf_utils.cust_conv2d(out, 3, h_f=3, w_f=3, h_s=1, w_s=1, batch_norm=False, scope_name="out17",
                                       activation_fn=tf.tanh)
            return out

    def compute_loss(self, D_real_in, D_real_out, D_gen_in, D_gen_out, k_t, gamma=0.75):
        def autoencoder_loss(out, inp, eta=1):
            diff = tf.abs(out - inp)
            return tf.reduce_sum(tf.pow(diff, eta))

        mu_real = autoencoder_loss(D_real_out, D_real_in)
        mu_gen = autoencoder_loss(D_gen_out, D_gen_in)
        D_loss = mu_real - k_t * mu_gen
        G_loss = mu_gen
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

            self.kl_loss = -self._log_sigma + 0.5 * (-1 + tf.exp(2 * self._log_sigma) + tf.square(self._mean))
            self.kl_loss = tf.reduce_mean(self.kl_loss)
            g_loss += self.kl_loss

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
        tf.summary.image(name="gen_imgae", tensor=self.gen_images, max_outputs=num_images)

        tf.summary.image(name="reconstructed_true_image", tensor=self.reconstructed_true_image, max_outputs=num_images)
        tf.summary.image(name="reconstructed_gen_image", tensor=self.reconstructed_gen_image, max_outputs=num_images)

        # Add summaries for loss functions
        tf.summary.scalar(name="kl_loss", tensor=self.kl_loss)
        tf.summary.scalar(name="g_loss", tensor=self.g_loss)
        tf.summary.scalar(name="d_loss", tensor=self.d_loss)

        # Add summaries for exponential decaying variables
        tf.summary.scalar(name="learning_rate", tensor=self.learning_rate)

        self.merged_summary_op = tf.summary.merge_all()

    def train(self):
        self.saver, self.summary_writer = helper.restore(self)

        with self.sess.as_default() as sess:
            self.global_step.eval()

            tf.train.start_queue_runners(sess=self.sess)
            coord = tf.train.Coordinator()

            nb_epochs = self.nb_epochs
            save_every = 3
            summary_every = 50
            nb_train_iter = 1000
            decay_every = 50
            k_t_ = 0

            for epoch in trange(nb_epochs, desc="Epoch"):
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