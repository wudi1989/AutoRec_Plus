import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from utils.loss_function import l2_norm, l1_l2_loss
from utils.trainer import train_model
from utils.tester import test_model
from utils.saver import make_records, save_model
import time
import numpy as np
import math


class AutoRec():
    def __init__(self, args, num_users, num_items, train_R, train_mask_R, test_R, test_mask_R, num_train_ratings,
                 num_test_ratings, user_train_set, item_train_set, user_test_set, item_test_set):

        self.args = args
        self.save_model = args.save_model

        self.num_users = num_users
        self.num_items = num_items

        self.train_R = train_R
        self.train_mask_R = train_mask_R
        self.test_R = test_R
        self.test_mask_R = test_mask_R
        self.num_train_ratings = num_train_ratings
        self.num_test_ratings = num_test_ratings

        self.user_train_set = user_train_set
        self.item_train_set = item_train_set
        self.user_test_set = user_test_set
        self.item_test_set = item_test_set

        self.hidden_neuron = args.hidden_neuron
        self.train_epoch = args.train_epoch
        self.batch_size = args.batch_size
        self.num_batch = int(math.ceil(self.num_items / float(self.batch_size)))

        self.base_lr = args.base_lr
        self.optimizer_method = args.optimizer_method
        self.display_step = args.display_step
        self.random_seed = args.random_seed
        self.isL1L2 = args.isL1L2
        self.global_step = tf.Variable(0, trainable=False)
        self.decay_epoch_step = args.decay_epoch_step
        self.decay_step = self.decay_epoch_step * self.num_batch
        self.lr = tf.train.exponential_decay(self.base_lr, self.global_step,
                                             self.decay_step, 0.8, staircase=True)
        self.lambda_value = args.lambda_value

        self.train_cost_list = []
        self.test_cost_list = []
        self.test_rmse_list = []
        self.test_mae_list = []
        self.time_list = []
        self.grad_clip = args.grad_clip

        self.Encoder = None

        self.gamma = tf.sqrt(1 / tf.log(np.float(args.train_epoch)))

        self.cul_l1 = 0
        self.cul_l2 = 0

    def prepare_model(self, w_ci, w_cu):

        dataset = tf.data.Dataset.from_tensor_slices((self.train_R, self.train_mask_R, list(self.item_train_set),
                                                      self.PB_bi, self.PB_bu, self.test_R, self.test_mask_R))
        train_dataset = dataset.shuffle(buffer_size=1000)
        train_dataset = train_dataset.batch(self.batch_size)

        test_dataset = dataset.batch(self.batch_size)
        iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
        self.train_init_op = iter.make_initializer(train_dataset)
        self.test_init_op = iter.make_initializer(test_dataset)

        input_R, input_mask_R, input_ids, input_PB_bi, input_PB_bu, input_test_r, input_test_r_mask = iter.get_next()
        input_R, input_mask_R, input_PB_bi, input_PB_bu, input_test_r, input_test_r_mask = tf.cast(input_R,
                                                                                                   tf.float32), tf.cast(
            input_mask_R,
            tf.float32), tf.cast(
            input_PB_bi, tf.float32), tf.cast(input_PB_bu, tf.float32), tf.cast(input_test_r, tf.float32), tf.cast(
            input_test_r_mask, tf.float32)
        V = tf.get_variable(name="V", initializer=tf.truncated_normal(shape=[self.num_users, self.hidden_neuron],
                                                                      mean=0, stddev=0.01), dtype=tf.float32)
        W = tf.get_variable(name="W", initializer=tf.truncated_normal(shape=[self.hidden_neuron, self.num_users],
                                                                      mean=0, stddev=0.01), dtype=tf.float32)
        mu = tf.get_variable(name="mu", initializer=tf.random_normal(shape=[self.hidden_neuron], stddev=0.01),
                             dtype=tf.float32)
        b = tf.get_variable(name="b", initializer=tf.random_normal(shape=[self.num_users], stddev=0.01),
                            dtype=tf.float32)

        TB_u_par = tf.get_variable(name="tb_u", initializer=tf.ones(shape=self.num_users), dtype=tf.float32)
        TB_i_par = tf.get_variable(name="tb_i", initializer=tf.ones(shape=self.num_items), dtype=tf.float32)
        TB_i = tf.gather(TB_i_par, input_ids)
        pre_Encoder = tf.matmul(input_R, V) + mu
        self.Encoder = tf.nn.sigmoid(pre_Encoder)
        pre_Decoder = tf.matmul(self.Encoder, W) + b
        pre_Decoder = tf.add(tf.add(tf.add(tf.add(tf.add(pre_Decoder, TB_u_par * w_cu), tf.reshape(TB_i * w_ci, (
            tf.shape(input_R)[0], 1))), input_PB_bi), input_PB_bu), self.PB_mu)
        self.Decoder = tf.identity(pre_Decoder)

        pre_numerator = tf.multiply((pre_Decoder - input_test_r), input_test_r_mask)
        self.numerator_mae = tf.reduce_sum(tf.abs(pre_numerator))
        self.numerator_rmse = tf.reduce_sum(tf.square(pre_numerator))

        pre_rec_cost = tf.multiply((input_R - self.Decoder), input_mask_R)

        rec_cost = 0

        if self.isL1L2:
            rec_cost, l1_cost, l2_cost = l1_l2_loss(pre_rec_cost, self.gamma, self.cul_l1, self.cul_l2)
        else:
            rec_cost = tf.square(l2_norm(pre_rec_cost))

        pre_reg_cost = tf.square(l2_norm(W)) + tf.square(l2_norm(V)) + tf.square(
            tf.reduce_sum(TB_i)) * w_ci + tf.square(tf.reduce_sum(TB_u_par)) * w_cu
        reg_cost = self.lambda_value * pre_reg_cost

        self.cost = rec_cost + reg_cost

        if self.optimizer_method == "Adam":
            optimizer = tf.train.AdamOptimizer(self.lr)
        elif self.optimizer_method == "RMSProp":
            optimizer = tf.train.RMSPropOptimizer(self.lr)
        else:
            raise ValueError("Optimizer Key ERROR")

        if self.grad_clip:
            gvs = optimizer.compute_gradients(self.cost)
            capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
            self.optimizer = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
        else:
            self.optimizer = optimizer.minimize(self.cost, global_step=self.global_step)

    def run(self, sess, result_path, model_path, PB_mu, PB_bi, PB_bu, w_ci, w_cu):
        self.sess = sess
        self.result_path = result_path
        self.PB_bi = PB_bi
        self.PB_bu = PB_bu
        self.PB_mu = PB_mu

        self.prepare_model(w_ci, w_cu)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        total_time = 0
        min_rmse = 65535
        counter = 0
        for epoch_itr in range(self.train_epoch):
            start_time = time.time()
            # train
            self.train_cost_list.append(
                train_model(self.sess, self.train_init_op, self.num_batch, self.optimizer, self.cost, epoch_itr,
                            self.display_step))
            # test
            costs, RMSE, MAE = test_model(self.sess, self.test_init_op, self.num_batch, self.cost,
                                          self.num_test_ratings, self.display_step, self.numerator_rmse,
                                          self.numerator_mae, epoch_itr)

            if self.save_model and counter % 10 == 0:
                counter += 1
                if min_rmse > RMSE:
                    min_rmse = RMSE
                    save_model(sess, model_path)

            self.test_cost_list.append(costs)
            self.test_rmse_list.append(RMSE)
            self.test_mae_list.append(MAE)

            total_time += (time.time() - start_time)
            self.time_list.append(total_time)
        make_records(self.args, self.result_path, self.test_rmse_list, self.test_mae_list, self.train_cost_list,
                     self.test_cost_list, self.time_list)
        return self.test_rmse_list, min(self.test_rmse_list), str(
            self.test_rmse_list.index(min(self.test_rmse_list))), min(self.test_mae_list), str(
            self.test_mae_list.index(min(self.test_mae_list)))

    def get_encoder(self, sess, result_path, PB_mu, PB_bi, PB_bu, w_mu, w_bi, w_bu, w_ci, w_cu):
        self.sess = sess
        self.result_path = result_path
        self.PB_bi = PB_bi * w_bi
        self.PB_bu = PB_bu * w_bu
        self.PB_mu = PB_mu * w_mu
        self.prepare_model(w_ci, w_cu)
        saver = tf.train.Saver()
        saver.restore(self.sess, result_path + "model.ckpt")
        self.sess.run(self.test_init_op)
        total_encoder_mx = []
        decoder_list = []
        costs = 0
        for i in range(self.num_batch):
            Cost, Encoder, Decoder = self.sess.run([self.cost, self.Encoder, self.Decoder])
            total_encoder_mx.append(Encoder)
            decoder_list.append(Decoder)
            costs += Cost
        self.test_cost_list.append(costs)
        Decoder = np.concatenate(decoder_list)
        print(Decoder.shape)

        Estimated_R = Decoder

        pre_numerator = np.multiply((Estimated_R - self.test_R), self.test_mask_R)
        mae_pre_numerator = np.abs(pre_numerator)
        mae_numerator = np.sum(mae_pre_numerator)

        numerator = np.sum(np.square(pre_numerator))
        denominator = self.num_test_ratings
        RMSE = np.sqrt(numerator / float(denominator))
        MAE = mae_numerator / float(denominator)
        self.test_mae_list.append(MAE)
        self.test_rmse_list.append(RMSE)
        # save model

        print("Testing //", " Total cost = {:.2f}".format(costs),
              " RMSE = {:.5f}".format(RMSE), " MAE = {:.5f}".format(MAE))
        Encoder = np.concatenate(total_encoder_mx)
        return Encoder
