from decimal import Decimal

import numpy as np
import tensorflow as tf
from utils.data_preprocessor import *
from utils.Preprossing_bias import getPB
from models.AutoRec_pp import AutoRec
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import time
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
current_time = time.time()

parser = argparse.ArgumentParser(description='AutoRec_pp')

parser.add_argument('--hidden_neuron', type=int, default=500)
parser.add_argument('--lambda_value', type=float, default=10)

parser.add_argument('--train_epoch', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=700)

parser.add_argument('--optimizer_method', choices=['Adam', 'RMSProp'], default='Adam')
parser.add_argument('--grad_clip', type=bool, default=False)
parser.add_argument('--base_lr', type=float, default=1e-3)
parser.add_argument('--decay_epoch_step', type=int, default=50, help="decay the learning rate for each n epochs")

parser.add_argument('--random_seed', type=int, default=1000)
parser.add_argument('--display_step', type=int, default=1)

parser.add_argument('--isL1L2', type=bool, default=True,
                    help="choose the loss function, False as the l2 loss, Ture as the L1-L2 loss")

parser.add_argument('--save_model', type=bool, default=False)

parser.add_argument('--data_name', choices=["Ml1M", "Ml100k", "Hetrec-ML", "Yahoo", "douban"], default='Ml100k')

args = parser.parse_args()
tf.set_random_seed(args.random_seed)
np.random.seed(args.random_seed)

data_name_list = ["Ml1M", "Ml100k", "Hetrec-ML", "Yahoo", "douban"]
num_users_list = [6040, 943, 2113, 15400, 3000]
num_items_list = [3952, 1682, 10109, 1000, 3000]
num_total_ratings_list = [1000209, 100000, 855598, 365704, 136891]
learning_rate_list = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3]
reg_rate_list = [20, 20, 20, 10, 10]
batch_size_list = [1800, 700, 4500, 700, 1500]
data_splits = ["80-20", "50-50", "20-80"]

test_rmse_list, test_mae_list, rmse_ep_list, mae_ep_list = [], [], [], []

i = data_name_list.index(args.data_name)

data_name = data_name_list[i]
num_users = num_users_list[i]
num_items = num_items_list[i]
num_total_ratings = num_total_ratings_list[i]

args.base_lr = learning_rate_list[i]
args.lambda_value = reg_rate_list[i]
args.batch_size = batch_size_list[i]
train_ratio = 0.9

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

best_rmse_result_list = []
best_mae_result_list = []

PB_weight = [1, 1, 1]
TB_weight = [1, 1]

w_ci, w_cu = TB_weight
w_mu, w_bi, w_bu = PB_weight
print(w_mu, w_bi, w_bu, w_ci, w_cu)

for j in range(len(data_splits)):
    data_split = data_splits[j]
    print("data_split:" + data_split)

    path = "data/%s" % data_split + "/%s" % data_name + "/"

    train_R, train_mask_R, test_R, test_mask_R, num_train_ratings, num_test_ratings, \
    user_train_set, item_train_set, user_test_set, item_test_set, num_Ru, num_Ri \
        = read_rating(path, num_users, num_items, num_total_ratings, 1, 0, train_ratio)

    # pb
    PB_mu, PB_bi, PB_bu = getPB(train_R, train_mask_R, num_Ri, num_Ru, w_mu, w_bi, w_bu, num_train_ratings)

    result_path = 'results/' + data_name + '/' + data_split + '_' + str(args.optimizer_method) + '_' + str(
        args.base_lr) + "_" + str(args.lambda_value) + "/"

    model_path = 'results/models/' + data_name + '/' + data_split + '_' + str(args.optimizer_method) + '_' + str(
        args.base_lr) + "_" + str(args.lambda_value) + "/"

    with tf.Session(config=config) as sess:
        Autorec = AutoRec(args, num_users, num_items, train_R.T, train_mask_R.T, test_R.T, test_mask_R.T,
                          num_train_ratings, num_test_ratings, user_train_set, item_train_set, user_test_set,
                          item_test_set)
        # run
        test_result, best_RMSE_result, epoch_RMSE_index, best_MAE_result, epoch_MAE_index = Autorec.run(sess,
                                                                                                        result_path,
                                                                                                        model_path,
                                                                                                        PB_mu,
                                                                                                        PB_bi.T,
                                                                                                        PB_bu.T,
                                                                                                        w_ci,
                                                                                                        w_cu)
        best_mae_result_list.append(str(Decimal(best_MAE_result).quantize(Decimal('0.000'))))
        best_rmse_result_list.append(str(Decimal(best_RMSE_result).quantize(Decimal('0.000'))))
        print("=========best_RMSE_result:" + str(
            Decimal(best_RMSE_result).quantize(Decimal('0.000'))) + "best_MAE_result:" + str(
            Decimal(best_MAE_result).quantize(Decimal('0.000'))) + "============")
        # Autorec.save_model(model_result_path)

        test_rmse_list.append(str(best_RMSE_result))
        test_mae_list.append(str(best_MAE_result))
        rmse_ep_list.append(str(epoch_RMSE_index))
        mae_ep_list.append(str(epoch_MAE_index))

        sess.close()
    tf.reset_default_graph()

for i in range(len(data_splits)):
    result = "|split-ratio|" + data_splits[i] + "|" + "RMSE|" + test_rmse_list[i] + "|" + "MAE|" + test_mae_list[
        i] + "|"
    print(result.replace("'", "|"))
