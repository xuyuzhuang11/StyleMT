# usage: python build_datastore.py luxun all-10 mobi-all10

import numpy as np
import sys
import os

if __name__ == "__main__":
    print("Start building datastores of layers...")
    feature_dim = 512
    # feature_dim = 1024
    folder_name = ["mono.0", "mono.1", "mono.2", "mono.3", "mono.4", "mono.5", "bili.0", "bili.1", "bili.2", "bili.3", "bili.4", "bili.5"]
    path = "./" + sys.argv[2]
    save_path = "./" + sys.argv[1] + "-datastore/" + sys.argv[3]
    os.makedirs(save_path, exist_ok=True)
    folder_length = []
    print("  calculating number of files in different layers...")
    for name in ["mono.0", "mono.1", "mono.2", "mono.3", "mono.4", "mono.5"]:
        tmp_path = path + "/" + name
        folder_length.append(len(os.listdir(tmp_path)) // 2)
    print("  ", end="")
    print(folder_length)


    for i in range(0, 6):
        print(f"  layer {i}")
        for folder in [folder_name[i], folder_name[i + 6]]:
            # print(folder)

            keys = np.zeros((folder_length[i], feature_dim))
            for j in range(0, folder_length[i]):                       # number of keys, must choose correctly
                if folder[0] == "m":
                    key = np.load(path + "/" + folder + "/" + str(j + 1) + ".v.npy", allow_pickle=True)
                else:
                    key = np.load(path + "/" + folder + "/" + str(j + 1) + ".k.npy", allow_pickle=True)
                # key = np.load(path + "/" + folder + "/" + str(j + 1) + ".k.npy", allow_pickle=True)
                keys[j] = key
                if j % 10000 == 0:
                    print("  " + str(j))
            print("  " + str(keys.shape))

            values = np.zeros((folder_length[i], feature_dim))
            if folder[0] == "b" or folder[0] == "m":
                values = keys
                print("  " + str(values.shape))
                np.savez(save_path + "/" + folder + ".npz", keys=keys, values=values)
                continue
            for j in range(0, folder_length[i]):  # number of keys, must choose correctly
                value = np.load(path + "/" + folder + "/" + str(j + 1) + ".v.npy", allow_pickle=True)
                values[j] = value
                if j % 10000 == 0:
                    print("  " + str(j))
            print("  " + str(values.shape))

            np.savez(save_path + "/" + folder + ".npz", keys=keys, values=values)
