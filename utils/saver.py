import os
import tensorflow as tf

def make_records(args, result_path, test_rmse_list, test_mae_list, train_cost_list, test_cost_list, time_list):
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    basic_info = result_path + "basic_info.txt"
    train_record = result_path + "train_record.txt"
    test_record = result_path + "test_record.txt"

    rmse_time = str(time_list[int(test_rmse_list.index(min(test_rmse_list)))])
    mae_time = str(time_list[int(test_mae_list.index(min(test_mae_list)))])

    print("rmse_time:" + rmse_time + ";mae_time:" + mae_time)
    with open(train_record, 'w') as f:
        f.write(str("Cost:"))
        f.write('\t')
        for itr in range(len(train_cost_list)):
            f.write(str(train_cost_list[itr]))
            f.write('\t')
        f.write('\n')

    with open(test_record, 'w') as g:
        g.write(str("Cost:"))
        g.write('\t')
        for itr in range(len(test_cost_list)):
            g.write(str(test_cost_list[itr]))
            g.write('\t')
        g.write('\n')

        g.write(str("RMSE:"))
        for itr in range(len(test_rmse_list)):
            g.write(str(test_rmse_list[itr]))
            g.write('\t')
        g.write('\n')

        g.write(str("MAE:"))
        for itr in range(len(test_mae_list)):
            g.write(str(test_mae_list[itr]))
            g.write('\t')
        g.write('\n')

        g.write(str("Best_RMSE:"))
        g.write(str(min(test_rmse_list)))
        g.write('\n')

        g.write(str("Best_RMSE_epoch:"))
        g.write(str(test_rmse_list.index(min(test_rmse_list))))
        g.write('\n')

        g.write(str("Best_RMSE_time:"))
        g.write(rmse_time)
        g.write('\n')

        g.write(str("Best_MAE:"))
        g.write(str(min(test_mae_list)))
        g.write('\n')

        g.write(str("Best_MAE_epoch:"))
        g.write(str(test_mae_list.index(min(test_mae_list))))
        g.write('\n')

        g.write(str("Best_MAE_time:"))
        g.write(mae_time)
        g.write('\n')

    with open(basic_info, 'w') as h:
        h.write(str(args))


def save_model(sess, model_path):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    saver = tf.train.Saver()
    saver.save(sess, model_path + "model.ckpt")