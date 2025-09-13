import os
import shutil
import uuid
import imghdr
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
RAW_DIR = "raw_dataset"         # original dataset folder
CLEAN_DIR = "dataset_clean"     # cleaned + restructured
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
SEED = 42
SUPPORTED = {"jpeg", "png", "bmp"}  # imghdr.what() names
SPLIT = [0.7, 0.15, 0.15]      # train / val / test split

ImageFile.LOAD_TRUNCATED_IMAGES = False

# -----------------------------
# STEP 1: CLEAN + RESTRUCTURE
# -----------------------------
def clean_and_restructure(raw_dir, clean_dir):
    if os.path.exists(clean_dir):
        shutil.rmtree(clean_dir)
    os.makedirs(clean_dir, exist_ok=True)

    bad_files = []
    saved_count = 0

    # Expect structure like all subfolders and images inside
    crops = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]

    for crop in crops:
        crop_path = os.path.join(raw_dir, crop)
        dest_crop = os.path.join(clean_dir, crop)
        os.makedirs(dest_crop, exist_ok=True)

        for root, _, files in os.walk(crop_path):
            for fname in files:
                src = os.path.join(root, fname)
                try:
                    if os.path.getsize(src) == 0:
                        bad_files.append((src, "zero size"))
                        continue

                    kind = imghdr.what(src)
                    if kind is None or kind.lower() not in SUPPORTED:
                        bad_files.append((src, f"unsupported kind={kind}"))
                        continue

                    with Image.open(src) as im:
                        im.verify()
                    with Image.open(src) as im:
                        im = im.convert("RGB")
                        dest_filename = f"{uuid.uuid4().hex}.jpg"
                        dest_path = os.path.join(dest_crop, dest_filename)
                        im.save(dest_path, format="JPEG", quality=90)
                        saved_count += 1

                except Exception as e:
                    bad_files.append((src, str(e)))

    print(f"✅ Cleaned and saved {saved_count} images to {clean_dir}")
    print(f"❌ Skipped {len(bad_files)} bad files")
    if bad_files:
        print("Example bad files:", bad_files[:5])


clean_and_restructure(RAW_DIR, CLEAN_DIR)

# -----------------------------
# STEP 2: SPLIT into train/val/test
# -----------------------------
def split_dataset(clean_dir, split_ratio):
    final_base = "dataset_final"
    if os.path.exists(final_base):
        shutil.rmtree(final_base)
    os.makedirs(final_base, exist_ok=True)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(final_base, split), exist_ok=True)

    crops = [d for d in os.listdir(clean_dir) if os.path.isdir(os.path.join(clean_dir, d))]
    for crop in crops:
        files = [os.path.join(clean_dir, crop, f) for f in os.listdir(os.path.join(clean_dir, crop))]
        train_files, testval = train_test_split(files, test_size=1 - split_ratio[0], random_state=SEED)
        val_files, test_files = train_test_split(testval, test_size=split_ratio[2] / sum(split_ratio[1:]), random_state=SEED)

        for split, subset in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            dest = os.path.join(final_base, split, crop)
            os.makedirs(dest, exist_ok=True)
            for f in subset:
                shutil.copy(f, dest)

    return final_base


FINAL_DIR = split_dataset(CLEAN_DIR, SPLIT)
print("Dataset split into train/val/test inside:", FINAL_DIR)

# -----------------------------
# STEP 3: Load datasets
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(FINAL_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(FINAL_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(FINAL_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# -----------------------------
# STEP 4: Build Model (ResNet50)
# -----------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.06),
    layers.RandomZoom(0.08),
])

IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=IMG_SHAPE)
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = tf.keras.applications.resnet.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation="relu")(x)

if num_classes == 2:
    outputs = layers.Dense(1, activation="sigmoid")(x)
    loss = "binary_crossentropy"
else:
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    loss = "sparse_categorical_crossentropy"

model = models.Model(inputs, outputs)
model.compile(optimizer=optimizers.Adam(1e-4), loss=loss, metrics=["accuracy"])
model.summary()

# -----------------------------
# STEP 5: Train
# -----------------------------
checkpoint = ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True)
early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)

EPOCHS = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, early, reduce_lr]
)

# -----------------------------
# STEP 6: Plot curves
# -----------------------------
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend(); plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend(); plt.title("Loss")
plt.show()

# -----------------------------
# STEP 7: Save final model
# -----------------------------
model.save("pest_model_final.keras")
print("✅ Model saved as pest_model_final.keras and best_model.h5")

# -----------------------------
# STEP 8: Evaluate on test set
# -----------------------------
test_loss, test_acc = model.evaluate(test_ds)
print(f"📊 Test Accuracy: {test_acc:.4f}")

