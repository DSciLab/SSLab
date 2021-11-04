import random
import copy
from typing import Any, List, Tuple, Union

from numpy import number


def split_by_proportion(
    meta_data,
    proportions: Union[List[float], Tuple[float, float]]
) -> Tuple[List[Any]]:
    if isinstance(proportions, float):
        proportions = [proportions, 1 - proportions]
    assert proportions[0] < 1.0 or proportions[0] > 0.0
    assert proportions[1] < 1.0 or proportions[1] > 0.0

    real_images = meta_data['real_images']
    fake_images = meta_data['fake_images']
    
    random.shuffle(real_images)
    random.shuffle(fake_images)

    real_len = len(real_images)
    fake_len = len(fake_images)

    real_training_len = int(proportions[0] * real_len)
    fake_training_len = int(proportions[0] * fake_len)

    real_to_training = real_images[:real_training_len]
    real_to_testing = real_images[real_training_len:]
    fake_to_training = fake_images[:fake_training_len]
    fake_to_testing = fake_images[fake_training_len:]

    training_set = real_to_training + fake_to_training
    testing_set = real_to_testing + fake_to_testing

    return training_set, testing_set


def split_by_k_fold(
    meta_data,
    k: int
) -> Tuple[List[Any]]:
    assert k > 0
    outputs = []
    real_images_folds = []
    fake_images_folds = []
    real_images = meta_data['real_images']
    fake_images = meta_data['fake_images']
    random.shuffle(real_images)
    random.shuffle(fake_images)
    real_len = len(real_images)
    fake_len = len(fake_images)

    number_of_real_per_fold = real_len // k
    number_of_fake_per_fold = fake_len // k

    def merge_list(lst):
        output = []
        for sub_lst in lst:
            output += sub_lst
        return output

    for i in range(k):
        if i == k - 1:
            real_images_folds.append(real_images)
            fake_images_folds.append(fake_images)
        else:
            real_curr_fold = real_images[:number_of_real_per_fold]
            real_images = real_images[number_of_real_per_fold:]
            fake_curr_fold = fake_images[:number_of_fake_per_fold]
            fake_images = fake_images[number_of_fake_per_fold:]

            real_images_folds.append(real_curr_fold)
            fake_images_folds.append(fake_curr_fold)

    for i in range(k):
        _real_images_folds = copy.deepcopy(real_images_folds)
        _fake_images_folds = copy.deepcopy(fake_images_folds)

        real_testing = copy.deepcopy(_real_images_folds[i])
        fake_testing = copy.deepcopy(_fake_images_folds[i])
        del _real_images_folds[i]
        del _fake_images_folds[i]
        real_training = merge_list(_real_images_folds)
        fake_training = merge_list(_fake_images_folds)

        testing_set = real_testing + fake_testing
        training_set = real_training + fake_training
        outputs.append((training_set, testing_set))

    return outputs
