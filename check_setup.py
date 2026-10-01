import tensorflow as tf
import os

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

data_dir = "data"
for split in ["train", "val", "test"]:
    for cls in ["NORMAL", "PNEUMONIA"]:
        path = os.path.join(data_dir, split, cls)
        print(path, len(os.listdir(path)))