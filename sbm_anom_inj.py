import pickle
import numpy as np
import random

with open('datasets/sbm/sbm_acts.pickle', 'rb') as handle:
    data = pickle.load(handle)

data = np.swapaxes(data, 0, 1)
print(data.shape)

static_data = data[:, :2000]

static_feats = static_data[1:]
static_lab_indices = static_data[0]

static_labels = np.zeros((static_lab_indices.shape[0], 10))


online_data = data[:, 2000:]

online_feats = online_data[1:]

online_len_dec = online_feats.shape[1] // 101

anom_nodes = [random.randint(0, 1000) for _ in range(100)]
print(anom_nodes)

anom_count = 100
anom_ones = np.ones((anom_count))

total_online_labels = np.zeros((online_len_dec))
total_online_feats = online_feats[:, :online_len_dec]

for i in range(100):
    anom_node = anom_nodes[i]
    targ_node = random.randint(0, 90)

    k = np.ones((3, anom_count))
    k[0] = anom_node
    k[1] = targ_node

    of_lb = online_len_dec * (i + 1)
    of_ub = online_len_dec * (i + 2)

    total_online_feats = np.concatenate((total_online_feats, k), 1)
    total_online_feats = np.concatenate((total_online_feats, online_feats[:,of_lb:of_ub]), 1)

    total_online_labels = np.concatenate((total_online_labels, anom_ones))
    total_online_labels = np.concatenate((total_online_labels, np.zeros((online_len_dec))))

print(total_online_feats.shape)
print(total_online_labels.shape)

return_data = [static_feats, static_labels, total_online_feats, total_online_labels]

pickle_dir = "datasets/sbm/sbm_anom.pickle"
with open(pickle_dir, 'wb') as handle:
    pickle.dump(return_data, handle, protocol=pickle.HIGHEST_PROTOCOL)