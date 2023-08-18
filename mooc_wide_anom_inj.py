import pickle
import numpy as np
import random

with open('datasets/act-mooc/mooc_action_pickle.pickle', 'rb') as handle:
    data = pickle.load(handle)

static_data = data[:, 4000:5000]

static_feats = np.concatenate((static_data[:2], np.expand_dims(static_data[8], 0)))
static_labels = static_data[3]

# online_data = data[:, 350000:]
online_data = data[:, 5000:40000]


online_feats = np.concatenate((online_data[:2], np.expand_dims(online_data[8], 0)))

# print("STATIC :", static_data.shape)
# print("ONLINE :", online_data.shape)

print(static_feats.shape, static_labels.shape, online_feats.shape)

online_len_dec = online_feats.shape[1] // 101

print("Online Len Dec", online_len_dec)

anom_nodes = [random.randint(0,500) for _ in range(100)]
print(anom_nodes)

anom_count = 100
anom_ones = np.ones((anom_count))

total_online_labels = np.zeros((online_len_dec))
total_online_feats = online_feats[:, :online_len_dec]

for i in range(100):
    anom_node = anom_nodes[i]

    k = np.ones((3, anom_count))
    k[0] = anom_node

    for j in range(anom_count):
        targ_node = random.randint(0, 90)
        k[1][j] = targ_node

    of_lb = online_len_dec * (i + 1)
    of_ub = online_len_dec * (i + 2)

    total_online_feats = np.concatenate((total_online_feats, k), 1)
    total_online_feats = np.concatenate((total_online_feats, online_feats[:,of_lb:of_ub]), 1)

    total_online_labels = np.concatenate((total_online_labels, anom_ones))
    total_online_labels = np.concatenate((total_online_labels, np.zeros((online_len_dec))))

print(total_online_feats.shape)
print(total_online_labels.shape)


return_data = [static_feats, static_labels, total_online_feats, total_online_labels]

pickle_dir = "datasets/act-mooc/mooc_action_anom.pickle"
with open(pickle_dir, 'wb') as handle:
    pickle.dump(return_data, handle, protocol=pickle.HIGHEST_PROTOCOL)