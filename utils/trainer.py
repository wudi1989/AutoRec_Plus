import time


def train_model(sess, train_init_op, num_batch, optimizer, cost, itr, display_step):
    start_time = time.time()
    sess.run(train_init_op)
    batch_cost = 0
    for i in range(num_batch):
        _, Cost = sess.run(
            [optimizer, cost])

        batch_cost = batch_cost + Cost

    if (itr + 1) % display_step == 0:
        print("Training //", "Epoch %d //" % (itr), " Total cost = {:.2f}".format(batch_cost),
              "Elapsed time : %d sec" % (time.time() - start_time))

    return batch_cost
