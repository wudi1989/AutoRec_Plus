import tensorflow as tf


def l2_norm(tensor):
    return tf.sqrt(tf.reduce_sum(tf.square(tensor)))


def l1_norm(tensor):
    return tf.reduce_sum(tf.abs(tensor))


def l1_l2_loss(pre_cost, gamma, cul_l1, cul_l2):
    alpha_1_numerator = tf.exp(-gamma * cul_l1)
    alpha_2_numerator = tf.exp(-gamma * cul_l2)
    alpha_1 = alpha_1_numerator / (alpha_1_numerator + alpha_2_numerator)
    alpha_2 = alpha_2_numerator / (alpha_1_numerator + alpha_2_numerator)

    l1_cost = l1_norm(pre_cost)
    l2_cost = tf.square(l2_norm(pre_cost))

    cost = alpha_1 * l1_cost + alpha_2 * l2_cost
    cul_l1 += l1_cost
    cul_l2 += l2_cost
    return cost, cul_l1, cul_l2
