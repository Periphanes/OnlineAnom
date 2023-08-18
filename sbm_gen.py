import random
import numpy as np
import pickle

com_cnt = 20
com_nd = 100

# 0 ~ 4 user -> 0 ~ 249
# 5 ~ 9 info -> 250 ~ 499

ln_prb = [[0 for j in range(com_cnt//2)] for i in range(com_cnt // 2)]
for i in range(com_cnt//2):
    ln_prb[i][i] = random.randint(600, 900)

    for j in range(com_cnt // 2):
        if j == i:
            continue
        else:
            ln_prb[i][j] = random.randint(0, 200)

lns = []

for i in range(com_cnt//2):
    for j in range(com_nd):
        for m in range(com_cnt//2):
            for n in range(com_nd):
                tar_num = m * com_nd + n
                str_num = i * com_nd + j
                ln_p = ln_prb[i][m]

                rnd_tmp = random.randint(0, 1000)
                if rnd_tmp <= ln_p:
                    rnd_dlt = random.randint(0, 100)
                    lns.append(np.array((str_num, tar_num, rnd_dlt)))

random.shuffle(lns)
ln = np.stack(lns)

print(ln.shape)
print(ln)

pickle_dir = "datasets/sbm/sbm_acts.pickle"
with open(pickle_dir, 'wb') as handle:
    pickle.dump(ln, handle, protocol=pickle.HIGHEST_PROTOCOL)

