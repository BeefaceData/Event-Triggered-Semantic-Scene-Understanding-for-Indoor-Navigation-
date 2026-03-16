import h5py
f = h5py.File("nyu_depth_v2_labeled.mat", "r")
print("images shape:", f["images"].shape)
print("keys:", list(f.keys()))
f.close()