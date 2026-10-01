import os
import shutil
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def merge_and_split(data_dir, seed=42, val_ratio=0.2):
    merged_dir = os.path.join(data_dir, "merged")
    train_dir = os.path.join(data_dir, "train_split")
    val_dir = os.path.join(data_dir, "val_split")

    if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)

    for cls in ["NORMAL", "PNEUMONIA"]:
        os.makedirs(os.path.join(train_dir, cls))
        os.makedirs(os.path.join(val_dir, cls))

        files = []
        for split in ["train", "val"]:
            folder = os.path.join(data_dir, split, cls)
            files += [os.path.join(folder, f) for f in os.listdir(folder)]

        random.seed(seed)
        random.shuffle(files)

        split_point = int(len(files) * (1 - val_ratio))
        train_files = files[:split_point]
        val_files = files[split_point:]

        for f in train_files:
            shutil.copy(f, os.path.join(train_dir, cls))
        for f in val_files:
            shutil.copy(f, os.path.join(val_dir, cls))

    return train_dir, val_dir


def get_generators(data_dir, img_size=150, batch_size=32):
    train_dir, val_dir = merge_and_split(data_dir)
    test_dir = os.path.join(data_dir, "test")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary"
    )

    val_gen = test_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False
    )

    return train_gen, val_gen, test_gen