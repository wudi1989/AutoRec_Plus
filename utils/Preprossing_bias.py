import numpy as np


def getPB(train_R, train_mask_R, num_Ri, num_Ru, w_mu, w_bi, w_bu, num_train_ratings):
    # PB
    PB_mu = train_R.sum() / num_train_ratings
    PB_bi = (((train_R - w_mu * PB_mu) * train_mask_R).sum(axis=0) / (
            train_R.sum(axis=0) / np.count_nonzero(train_R, axis=0) + num_Ri)).squeeze()

    # replace inf and nan with 0
    PB_bi[PB_bi == np.inf] = 0
    PB_bi[PB_bi == -np.inf] = 0
    PB_bi = np.nan_to_num(PB_bi)
    PB_bi = np.repeat(np.expand_dims(PB_bi, axis=0), train_R.shape[0], axis=0)
    PB_bu = (((train_R - w_mu * PB_mu - PB_bi * w_bi) * train_mask_R).sum(axis=1) / (
            train_R.sum(axis=1) / np.count_nonzero(train_R, axis=1) + num_Ru)).squeeze()
    PB_bu[PB_bu == np.inf] = 0
    PB_bu[PB_bu == -np.inf] = 0
    PB_bu = np.nan_to_num(PB_bu)
    PB_bu = np.repeat(np.expand_dims(PB_bu, axis=1), train_R.shape[1], axis=1)

    return w_mu*PB_mu, w_bi*PB_bi, w_bu * PB_bu
