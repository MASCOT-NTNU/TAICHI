import numpy as np
import pickle

t = dict()
filehandler = open("test.p", 'wb')
with filehandler as f:
    pickle.dump(t, f)
f.close()

for i in range(50):
    print(i)
    tmp = np.random.randint(0, 40000, 20)
    t[i] = tmp

    neighbour_file = open("test.p", 'rb')
    new_t = pickle.load(neighbour_file)
    neighbour_file.close()

    new_t[i] = tmp

    f = open("test.p", 'wb')
    pickle.dump(new_t, f)
    f.close()
    # if i > 10:
    #     break

#%%

neighbour_file = open("test.p", 'rb')
new_t = pickle.load(neighbour_file)
neighbour_file.close()

#%%
for j in range(len(t)):
    print(np.all(t[j] == new_t[j]))

