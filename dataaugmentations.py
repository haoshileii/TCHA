import numpy as np
import torch

def up_sample(data):
    length = data.shape[0]
    insert_data = [(data[i] + data[i + 1]) / 2.0 for i in range(0, length - 1, 2)]
    up_sample_series = []
    k = 0
    for j in range(int(length / 2) * 2):
        up_sample_series.append(data[j])
        if j % 2 == 0:
            up_sample_series.append(insert_data[k])
            k += 1
    if length % 2 != 0:
        up_sample_series.append(data[-1])
    return up_sample_series

def down_sample(data):
    length = data.shape[0]
    if length % 2 == 0:
        return [(data[i] + data[i + 1]) / 2.0 for i in range(0, length - 1, 2)]
    else:
        down_sample_series = [(data[i] + data[i + 1]) / 2.0 for i in range(0, length - 1, 2)]
        down_sample_series.append(data[length - 1])
        return down_sample_series

#origin_series:array
def generate_series(origin_series):
    series_length = origin_series.shape[0]#the length of subsequence
    half = series_length / 2
    down_sample_series = down_sample(origin_series[:int(half)])
    if series_length % 2 == 0:
        up_sample_series = up_sample(origin_series[int(half):])
        series = np.append(down_sample_series, up_sample_series, axis=0)
        return series
    else:
        up_sample_series = up_sample(origin_series[int(half): -1])
        series = np.append(down_sample_series, up_sample_series, axis=0)
        series = np.concatenate((series, origin_series[-1].reshape(1, -1)), axis=0)
        return series

# origin_data: array
def get_target_series(slices, origin_data):
    sample_nums = origin_data.shape[0]#the number of initial samples
    length = origin_data.shape[1]#the length of initial samples
    dimensions = origin_data.shape[-1]
    part_length = int(length / slices) * 2
    target_series = []
    for j in range(int(slices / 2) - 1):
        target_series.append(np.array([generate_series(origin_data[i][j * part_length:(j + 1) * part_length])
                                       for i in range(sample_nums)]).reshape(sample_nums, -1, dimensions))
    target_series.append(np.array([generate_series(origin_data[i][part_length * int((slices / 2 - 1)):])
                                   for i in range(sample_nums)]).reshape(sample_nums, -1, dimensions))
    return np.concatenate(target_series, axis=1)


#origin_data:tensor(8, 206, 3); origin_label:tensor(8); num_trans:4
def data_augmentation(origin_data, origin_label, num_trans):
    data = origin_data.cpu().numpy()
    label = origin_label.numpy()
    slice_list = [2, 4, 8, 16, 32, 64]
    sample_labels_list = []
    for i in range(num_trans):
        sample_labels_list.append(label)
    data_list = [data]
    for i in range(num_trans-1):
        version_target_series = get_target_series(slices=slice_list[i], origin_data=data)
        data_list.append(version_target_series)
    all_series = np.concatenate(data_list, axis=0)
    all_sample_labels = np.concatenate(sample_labels_list, axis=0)
    all_series = torch.tensor(all_series)
    all_sample_labels = torch.tensor(all_sample_labels)
    return all_series, all_sample_labels
#######################
#######################
#######################
#sample: tensor(num,length,diention),labels: tensor(num)
def data_augmentationWR(sample):

    weak_aug = scaling(sample)
    strong_aug = jitter(permutation(sample))

    return strong_aug, weak_aug


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=6):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random" and x.shape[2] > 4:
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

