import numpy as np


def read_rating(path, num_users, num_items, num_total_ratings, a, b, train_ratio):
    # fp = open(path + "ratings-all.txt")

    user_train_set = set()
    user_test_set = set()
    item_train_set = set()
    item_test_set = set()
    fp_train = open(path + "train.txt")
    fp_test = open(path + "test.txt")

    lines_train = fp_train.readlines()
    lines_test = fp_test.readlines()

    num_Ri = np.zeros(num_items)
    num_Ru = np.zeros(num_users)
    train_R = np.zeros((num_users, num_items))
    train_R_list = []
    test_R = np.zeros((num_users, num_items))

    train_mask_R = np.zeros((num_users, num_items))
    test_mask_R = np.zeros((num_users, num_items))

    # random_perm_idx = np.random.permutation(num_total_ratings)
    # train_idx = random_perm_idx[0:int(num_total_ratings*train_ratio)]
    # test_idx = random_perm_idx[int(num_total_ratings*train_ratio):]

    num_train_ratings = len(lines_train)
    num_test_ratings = len(lines_test)

    ''' Train '''
    for line in lines_train:
        user, item, rating = line.split("::")
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        train_R[user_idx, item_idx] = float(rating.lstrip())
        train_mask_R[user_idx, item_idx] = 1

        user_train_set.add(user_idx)
        item_train_set.add(item_idx)

        train_R_list.append([user_idx, item_idx, float(rating.lstrip())])

        num_Ru[user_idx] = num_Ru[user_idx] + 1
        num_Ri[item_idx] = num_Ri[item_idx] + 1
    ''' Test '''
    for line in lines_test:
        user, item, rating = line.split("::")
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        test_R[user_idx, item_idx] = float(rating.lstrip())
        test_mask_R[user_idx, item_idx] = 1

        user_test_set.add(user_idx)
        item_test_set.add(item_idx)

        # num_Ru[user_idx] = num_Ru[user_idx] + 1
        # num_Ri[item_idx] = num_Ri[item_idx] + 1

    for i in range(num_items):
        item_train_set.add(i)
    return train_R, train_mask_R, test_R, test_mask_R, num_train_ratings, num_test_ratings, \
           user_train_set, item_train_set, user_test_set, item_test_set, num_Ru, num_Ri
