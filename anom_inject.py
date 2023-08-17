import pickle
import numpy as np

with open('datasets/act-mooc/mooc_action_pickle.pickle', 'rb') as handle:
    data = pickle.load(handle)

#ACT - MOOC

# Num Users : 7,047
# Num Targets : 97
# Num Actions 411,749
# Number of Positive Action Labels : 4,066
# Timestamp - Seconds

static_data = data[:, :350000]

static_feats = np.concatenate((static_data[:2], np.expand_dims(static_data[8], 0)))
static_labels = static_data[3]

online_data = data[:, 350000:]

online_feats = np.concatenate((online_data[:2], np.expand_dims(online_data[8], 0)))

# print("STATIC :", static_data.shape)
# print("ONLINE :", online_data.shape)

print(static_feats.shape, static_labels.shape, online_feats.shape)

online_len_dec = online_feats.shape[1] // 12

anom_nodes = [4854,934,7004,4031,6556,4324,524,643,1435,2343]

anom_ones = np.ones((300))

total_online_data = np.zeros((online_len_dec))

for i in range(10):
    anom_node = anom_nodes[i]
