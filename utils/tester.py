import time
import numpy as np

def test_model(sess, test_init_op, num_batch, cost, num_test_ratings, display_step, numerator_rmse, numerator_mae, itr):
    start_time = time.time()
    sess.run(test_init_op)
    numerator_rmse_ = 0
    numerator_mae_ = 0
    costs = 0
    for i in range(num_batch):
        Cost, num_rmse, num_mae = sess.run([cost, numerator_rmse, numerator_mae])
        costs += Cost
        numerator_rmse_ += num_rmse
        numerator_mae_ += num_mae
    RMSE = np.sqrt(numerator_rmse_ / float(num_test_ratings))
    MAE = numerator_mae_ / float(num_test_ratings)

    if (itr + 1) % display_step == 0:
        # save model

        print("Testing //", "Epoch %d //" % (itr), " Total cost = {:.2f}".format(costs),
              " RMSE = {:.5f}".format(RMSE), " MAE = {:.5f}".format(MAE),
              "Elapsed time : %d sec" % (time.time() - start_time))
        print("=" * 50)

    return costs, RMSE, MAE
