# =========================
# BLOCK 1: Environment Fix
# =========================

!pip uninstall -y albumentations albucore
!pip install albumentations==1.3.1 --no-cache-dir

import albumentations as A
print("Albumentations version:", A.__version__)
print("Has HorizontalFlip:", hasattr(A, "HorizontalFlip"))
print("Has VerticalFlip:", hasattr(A, "VerticalFlip"))


# =========================
# BLOCK 2: Imports & GPU
# =========================

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    jaccard_score,
    f1_score,
    precision_score,
    recall_score
)

from scipy.stats import ttest_rel

import albumentations as A
from einops import rearrange

# ---- GPU configuration ----
gpus = tf.config.list_physical_devices('GPU')
print("GPUs found:", gpus)

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print("GPU setup error:", e)


# =========================
# BLOCK 3: Dataset Paths & Classes
# =========================

# ---- Dataset base directory ----
data_dir = "../input/sdsa-dse-406-606-demo-data/"

train_images = os.path.join(data_dir, "train_images/train")
train_masks  = os.path.join(data_dir, "train_masks/train")

val_images = os.path.join(data_dir, "val_images/val")
val_masks  = os.path.join(data_dir, "val_masks/val")

test_images = os.path.join(data_dir, "test_images/test")
test_masks  = os.path.join(data_dir, "test_masks/test")

# ---- Sanity check paths ----
print("Train images exist:", os.path.exists(train_images))
print("Train masks exist :", os.path.exists(train_masks))
print("Val images exist  :", os.path.exists(val_images))
print("Val masks exist   :", os.path.exists(val_masks))
print("Test images exist :", os.path.exists(test_images))
print("Test masks exist  :", os.path.exists(test_masks))

# ---- Load class dictionary ----
class_df = pd.read_csv(os.path.join(data_dir, "class_dict.csv"))

label_names = list(class_df["name"])
r = class_df["r"].values
g = class_df["g"].values
b = class_df["b"].values

label_codes = [(ri, gi, bi) for ri, gi, bi in zip(r, g, b)]

code2id = {color: idx for idx, color in enumerate(label_codes)}
id2code = {idx: color for idx, color in enumerate(label_codes)}

num_classes = len(label_codes)

print("\nNumber of classes:", num_classes)
print("Class names:", label_names)


# =========================
# BLOCK 4: Mask Encoding Utilities
# =========================

def rgb_to_onehot(rgb_image, colormap):
    """
    Convert RGB mask to one-hot encoded mask
    """
    h, w, _ = rgb_image.shape
    onehot = np.zeros((h, w, len(colormap)), dtype=np.uint8)

    for idx, color in enumerate(colormap):
        onehot[:, :, idx] = np.all(rgb_image == color, axis=-1)

    return onehot


def onehot_to_rgb(onehot_mask, colormap):
    """
    Convert one-hot mask back to RGB
    """
    label_map = np.argmax(onehot_mask, axis=-1)
    rgb = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)

    for idx, color in enumerate(colormap):
        rgb[label_map == idx] = color

    return rgb


# ---- Quick sanity test ----
sample_mask_path = os.path.join(train_masks, os.listdir(train_masks)[0])

mask = cv2.imread(sample_mask_path)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

onehot = rgb_to_onehot(mask, label_codes)
reconstructed = onehot_to_rgb(onehot, label_codes)

print("Mask shape:", mask.shape)
print("One-hot shape:", onehot.shape)
print("Reconstruction identical:",
      np.all(mask == reconstructed))


# =========================
# BLOCK 5: Albumentations Augmentation
# =========================

transform = A.Compose(
    [
        A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
    ],
    is_check_shapes=False  # Important for mask safety in v2.x
)

# ---- Sanity test augmentation ----
sample_img_path = os.path.join(train_images, os.listdir(train_images)[0])
sample_mask_path = os.path.join(train_masks, os.listdir(train_masks)[0])

img = cv2.imread(sample_img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0

mask = cv2.imread(sample_mask_path)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

aug = transform(image=img, mask=mask)

print("Original image shape:", img.shape)
print("Augmented image shape:", aug["image"].shape)
print("Original mask unique colors:", np.unique(mask.reshape(-1, 3), axis=0).shape[0])
print("Augmented mask unique colors:", np.unique(aug["mask"].reshape(-1, 3), axis=0).shape[0])


# =========================
# BLOCK 6: Data Generator
# =========================

BATCH_SIZE = 16
IMG_SIZE = (256, 256)

def data_generator(
    image_dir,
    mask_dir,
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE,
    augment=False
):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".tif")])
    mask_files  = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

    assert len(image_files) == len(mask_files), "Image-mask count mismatch"

    while True:
        for start in range(0, len(image_files), batch_size):

            x_batch, y_batch = [], []

            for img_name, mask_name in zip(
                image_files[start:start + batch_size],
                mask_files[start:start + batch_size]
            ):
                img_path  = os.path.join(image_dir, img_name)
                mask_path = os.path.join(mask_dir, mask_name)

                # ---- Read image ----
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                img = img.astype(np.float32) / 255.0

                # ---- Read mask ----
                mask = cv2.imread(mask_path)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)

                # ---- Augmentation ----
                if augment:
                    augmented = transform(image=img, mask=mask)
                    img  = augmented["image"]
                    mask = augmented["mask"]

                # ---- One-hot encode mask ----
                mask_onehot = rgb_to_onehot(mask, label_codes)

                x_batch.append(img)
                y_batch.append(mask_onehot)

            yield np.array(x_batch), np.array(y_batch)


# ---- Generator sanity check ----
train_gen = data_generator(train_images, train_masks, augment=True)

x_sample, y_sample = next(train_gen)

print("Batch image shape:", x_sample.shape)
print("Batch mask shape :", y_sample.shape)
print("Image min/max    :", x_sample.min(), x_sample.max())
print("Mask unique sums :", np.unique(np.sum(y_sample, axis=-1)))


# =========================
# BLOCK 7: DeepLabV3+ Model
# =========================

def deeplabv3_plus(input_shape=(256, 256, 3), num_classes=7):
    inputs = layers.Input(shape=input_shape)

    # ---- Encoder: ResNet50 ----
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs
    )

    # High-level features
    x = base_model.get_layer("conv4_block6_out").output

    # ---- Atrous Spatial Pyramid Pooling (ASPP) ----
    def aspp(x):
        dims = x.shape

        y1 = layers.Conv2D(256, 1, padding="same", use_bias=False)(x)
        y1 = layers.BatchNormalization()(y1)
        y1 = layers.ReLU()(y1)

        y2 = layers.Conv2D(256, 3, dilation_rate=6, padding="same", use_bias=False)(x)
        y2 = layers.BatchNormalization()(y2)
        y2 = layers.ReLU()(y2)

        y3 = layers.Conv2D(256, 3, dilation_rate=12, padding="same", use_bias=False)(x)
        y3 = layers.BatchNormalization()(y3)
        y3 = layers.ReLU()(y3)

        y4 = layers.Conv2D(256, 3, dilation_rate=18, padding="same", use_bias=False)(x)
        y4 = layers.BatchNormalization()(y4)
        y4 = layers.ReLU()(y4)

        y5 = layers.GlobalAveragePooling2D()(x)
        y5 = layers.Reshape((1, 1, dims[-1]))(y5)
        y5 = layers.Conv2D(256, 1, padding="same", use_bias=False)(y5)
        y5 = layers.BatchNormalization()(y5)
        y5 = layers.ReLU()(y5)
        y5 = layers.UpSampling2D(
            size=(dims[1], dims[2]), interpolation="bilinear"
        )(y5)

        y = layers.Concatenate()([y1, y2, y3, y4, y5])
        y = layers.Conv2D(256, 1, padding="same", use_bias=False)(y)
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)

        return y

    x = aspp(x)

    # ---- Low-level features ----
    low_level = base_model.get_layer("conv2_block3_out").output
    low_level = layers.Conv2D(48, 1, padding="same", use_bias=False)(low_level)
    low_level = layers.BatchNormalization()(low_level)
    low_level = layers.ReLU()(low_level)

    # ---- Decoder ----
    x = layers.UpSampling2D(
        size=(4, 4), interpolation="bilinear"
    )(x)

    x = layers.Concatenate()([x, low_level])

    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.UpSampling2D(
        size=(4, 4), interpolation="bilinear"
    )(x)

    outputs = layers.Conv2D(
        num_classes, 1, activation="softmax"
    )(x)

    model = Model(inputs, outputs, name="DeepLabV3Plus")

    return model


# ---- Build model & summary ----
deeplab_model = deeplabv3_plus(
    input_shape=(256, 256, 3),
    num_classes=num_classes
)

deeplab_model.summary()


# =========================
# BLOCK 8: Losses & Metrics
# =========================

import tensorflow.keras.backend as K

# ---- Dice Coefficient ----
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = K.cast(y_true, "float32")
    y_pred = K.cast(y_pred, "float32")

    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true + y_pred, axis=[1, 2, 3])

    dice = (2. * intersection + smooth) / (union + smooth)
    return K.mean(dice)


# ---- Dice Loss ----
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


# ---- Categorical Focal Loss ----
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss(y_true, y_pred):
        y_true = K.cast(y_true, "float32")
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())

        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy

        return K.mean(K.sum(loss, axis=-1))
    return focal_loss


# ---- Combined Loss ----
def combined_dice_focal_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + categorical_focal_loss()(y_true, y_pred)


# ---- Metric: Mean IoU (soft) ----
def mean_iou(y_true, y_pred):
    # Cast BOTH to float32 (this fixes the error)
    y_true = tf.cast(y_true, tf.float32)

    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.one_hot(y_pred, depth=num_classes)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2]) - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    return tf.reduce_mean(iou)



# =========================
# BLOCK 9: Compile & Train (Baseline Sanity)
# =========================

# ---- Compile model ----
deeplab_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=combined_dice_focal_loss,
    metrics=[dice_coefficient, mean_iou]
)

# ---- Callbacks ----
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
]

# ---- Training steps ----
steps_per_epoch = len(os.listdir(train_images)) // BATCH_SIZE
val_steps = len(os.listdir(val_images)) // BATCH_SIZE

print("Steps per epoch:", steps_per_epoch)
print("Validation steps:", val_steps)

# =========================
# Validation Generator
# =========================

val_gen = data_generator(
    val_images,
    val_masks,
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE,
    augment=False
)

print("Validation generator created successfully")


# ---- Train (sanity epochs) ----
history = deeplab_model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)

# =========================
# BLOCK 10: Test Evaluation
# =========================

# ---- Create test generator ----
test_gen = data_generator(
    test_images,
    test_masks,
    batch_size=1,        # IMPORTANT: batch_size=1 for metrics
    img_size=IMG_SIZE,
    augment=False
)

num_test_samples = len(os.listdir(test_images))
print("Test samples:", num_test_samples)

# ---- Accumulators ----
dice_scores = []
iou_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# ---- Evaluation loop ----
for i in range(num_test_samples):
    x, y_true = next(test_gen)

    y_pred = deeplab_model.predict(x, verbose=0)

    # Convert to label maps
    y_true_lbl = np.argmax(y_true[0], axis=-1).flatten()
    y_pred_lbl = np.argmax(y_pred[0], axis=-1).flatten()

    dice_scores.append(
        f1_score(y_true_lbl, y_pred_lbl, average="macro", zero_division=0)
    )
    iou_scores.append(
        jaccard_score(y_true_lbl, y_pred_lbl, average="macro", zero_division=0)
    )
    precision_scores.append(
        precision_score(y_true_lbl, y_pred_lbl, average="macro", zero_division=0)
    )
    recall_scores.append(
        recall_score(y_true_lbl, y_pred_lbl, average="macro", zero_division=0)
    )
    f1_scores.append(
        f1_score(y_true_lbl, y_pred_lbl, average="macro", zero_division=0)
    )

# ---- Report ----
print("\n===== DeepLabV3+ Test Metrics =====")
print(f"Mean Dice      : {np.mean(dice_scores):.4f}")
print(f"Mean IoU       : {np.mean(iou_scores):.4f}")
print(f"Mean Precision : {np.mean(precision_scores):.4f}")
print(f"Mean Recall    : {np.mean(recall_scores):.4f}")
print(f"Mean F1-score  : {np.mean(f1_scores):.4f}")


# =========================
# BLOCK 11A: Visualization
# =========================

def visualize_predictions(model, image_dir, mask_dir, num_samples=3):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".tif")])
    mask_files  = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

    indices = np.random.choice(len(image_files), num_samples, replace=False)

    plt.figure(figsize=(12, 4 * num_samples))

    for i, idx in enumerate(indices):
        # ---- Load image ----
        img_path = os.path.join(image_dir, image_files[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img_norm = img.astype(np.float32) / 255.0

        # ---- Load mask ----
        mask_path = os.path.join(mask_dir, mask_files[idx])
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

        # ---- Predict ----
        pred = model.predict(img_norm[np.newaxis, ...], verbose=0)
        pred_rgb = onehot_to_rgb(pred[0], label_codes)

        # ---- Plot ----
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(img)
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(mask)
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(pred_rgb)
        plt.title("DeepLabV3+ Prediction")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# ---- Run visualization ----
visualize_predictions(
    deeplab_model,
    test_images,
    test_masks,
    num_samples=3
)

# =========================
# BLOCK 11B-FIX: SegFormer Baseline (Corrected)
# =========================

def mlp(x, hidden_dim, out_dim):
    x = layers.Dense(hidden_dim, activation="gelu")(x)
    x = layers.Dense(out_dim)(x)
    return x


def transformer_block(x, num_heads, embed_dim, mlp_ratio=4):
    h = layers.LayerNormalization(epsilon=1e-6)(x)

    h = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim // num_heads
    )(h, h)

    x = layers.Add()([x, h])

    h = layers.LayerNormalization(epsilon=1e-6)(x)
    h = mlp(h, embed_dim * mlp_ratio, embed_dim)

    return layers.Add()([x, h])


def patch_embedding(x, embed_dim, patch_size):
    x = layers.Conv2D(
        embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="same"
    )(x)
    h, w = x.shape[1], x.shape[2]
    x = layers.Reshape((h * w, embed_dim))(x)
    return x, h, w


def segformer_baseline(input_shape=(256, 256, 3), num_classes=7):
    inputs = layers.Input(shape=input_shape)

    # ---- Stage 1: 64x64 ----
    x, h1, w1 = patch_embedding(inputs, embed_dim=64, patch_size=4)
    for _ in range(2):
        x = transformer_block(x, num_heads=2, embed_dim=64)
    f1 = layers.Reshape((h1, w1, 64))(x)

    # ---- Stage 2: 32x32 ----
    x, h2, w2 = patch_embedding(f1, embed_dim=128, patch_size=2)
    for _ in range(2):
        x = transformer_block(x, num_heads=4, embed_dim=128)
    f2 = layers.Reshape((h2, w2, 128))(x)

    # ---- Stage 3: 16x16 ----
    x, h3, w3 = patch_embedding(f2, embed_dim=256, patch_size=2)
    for _ in range(4):
        x = transformer_block(x, num_heads=8, embed_dim=256)
    f3 = layers.Reshape((h3, w3, 256))(x)

    # ---- Align all features to 64x64 ----
    f1_up = f1
    f2_up = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(f2)
    f3_up = layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(f3)

    # ---- Fuse ----
    fused = layers.Concatenate()([f1_up, f2_up, f3_up])

    fused = layers.Conv2D(256, 1, padding="same")(fused)
    fused = layers.BatchNormalization()(fused)
    fused = layers.Activation("relu")(fused)

    # ---- Final upsampling to input resolution ----
    fused = layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(fused)

    outputs = layers.Conv2D(
        num_classes,
        kernel_size=1,
        activation="softmax"
    )(fused)

    return Model(inputs, outputs, name="SegFormer_Baseline")


# ---- Build model ----
segformer_model = segformer_baseline(
    input_shape=(256, 256, 3),
    num_classes=num_classes
)

segformer_model.summary()


# =========================
# BLOCK 11B.2: Compile & Train SegFormer
# =========================

# ---- Compile ----
segformer_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=combined_dice_focal_loss,
    metrics=[dice_coefficient, mean_iou]
)

# ---- Callbacks (reuse) ----
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
]

# ---- Train ----
history_segformer = segformer_model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)

# =========================
# BLOCK 11B.3: SegFormer Test Evaluation
# =========================

test_gen = data_generator(
    test_images,
    test_masks,
    batch_size=1,
    img_size=IMG_SIZE,
    augment=False
)

dice_scores = []
iou_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for _ in range(len(os.listdir(test_images))):
    x, y_true = next(test_gen)
    y_pred = segformer_model.predict(x, verbose=0)

    y_true_lbl = np.argmax(y_true[0], axis=-1).flatten()
    y_pred_lbl = np.argmax(y_pred[0], axis=-1).flatten()

    dice_scores.append(
        f1_score(y_true_lbl, y_pred_lbl, average="macro", zero_division=0)
    )
    iou_scores.append(
        jaccard_score(y_true_lbl, y_pred_lbl, average="macro", zero_division=0)
    )
    precision_scores.append(
        precision_score(y_true_lbl, y_pred_lbl, average="macro", zero_division=0)
    )
    recall_scores.append(
        recall_score(y_true_lbl, y_pred_lbl, average="macro", zero_division=0)
    )
    f1_scores.append(
        f1_score(y_true_lbl, y_pred_lbl, average="macro", zero_division=0)
    )

print("\n===== SegFormer Test Metrics =====")
print(f"Mean Dice      : {np.mean(dice_scores):.4f}")
print(f"Mean IoU       : {np.mean(iou_scores):.4f}")
print(f"Mean Precision : {np.mean(precision_scores):.4f}")
print(f"Mean Recall    : {np.mean(recall_scores):.4f}")
print(f"Mean F1-score  : {np.mean(f1_scores):.4f}")

# =========================
# BLOCK 11B.4: SegFormer Visualization
# =========================

def visualize_predictions_segformer(model, image_dir, mask_dir, num_samples=3):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".tif")])
    mask_files  = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

    indices = np.random.choice(len(image_files), num_samples, replace=False)

    plt.figure(figsize=(12, 4 * num_samples))

    for i, idx in enumerate(indices):
        # ---- Load image ----
        img_path = os.path.join(image_dir, image_files[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img_norm = img.astype(np.float32) / 255.0

        # ---- Load mask ----
        mask_path = os.path.join(mask_dir, mask_files[idx])
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

        # ---- Predict ----
        pred = model.predict(img_norm[np.newaxis, ...], verbose=0)
        pred_rgb = onehot_to_rgb(pred[0], label_codes)

        # ---- Plot ----
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(img)
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(mask)
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(pred_rgb)
        plt.title("SegFormer Prediction")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# ---- Run SegFormer visualization ----
visualize_predictions_segformer(
    segformer_model,
    test_images,
    test_masks,
    num_samples=3
)

# =========================
# BLOCK 12.1: InformalMamba CNN Stem
# =========================

def informal_cnn_stem(inputs):
    """
    Structure-preserving CNN stem.
    Designed to retain thin roads and slum boundaries.
    """

    x = layers.Conv2D(
        32, kernel_size=3, strides=1, padding="same"
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Edge-preserving depthwise conv
    x_dw = layers.DepthwiseConv2D(
        kernel_size=3, padding="same"
    )(x)
    x_dw = layers.BatchNormalization()(x_dw)

    x = layers.Add()([x, x_dw])
    x = layers.Activation("relu")(x)

    # Controlled downsampling (NO aggressive pooling)
    x = layers.Conv2D(
        64, kernel_size=3, strides=2, padding="same"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x

# =========================
# BLOCK 12.2: Directional Spatial State Scan
# =========================

class DirectionalStateScan(layers.Layer):
    """
    Converts a 2D feature map into directional state-space sequences
    and reconstructs spatial features from learned state trajectories.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # State projection
        self.state_proj = layers.Dense(channels)

        # Reset gate projection (used later, but defined here)
        self.gate_proj = layers.Dense(channels, activation="sigmoid")

    def scan_sequence(self, x, reverse=False):
        """
        x: (B, T, C)
        """
        if reverse:
            x = tf.reverse(x, axis=[1])

        states = []
        s = tf.zeros_like(x[:, 0, :])

        for t in range(x.shape[1]):
            xt = x[:, t, :]

            gate = self.gate_proj(xt)
            candidate = self.state_proj(xt)

            s = gate * s + (1.0 - gate) * candidate
            states.append(s)

        states = tf.stack(states, axis=1)

        if reverse:
            states = tf.reverse(states, axis=[1])

        return states

    def call(self, x):
        """
        x: (B, H, W, C)
        """
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        # ---- Horizontal scans ----
        x_lr = tf.reshape(x, (B * H, W, C))
        h_lr = self.scan_sequence(x_lr, reverse=False)
        h_lr = tf.reshape(h_lr, (B, H, W, C))

        h_rl = self.scan_sequence(x_lr, reverse=True)
        h_rl = tf.reshape(h_rl, (B, H, W, C))

        # ---- Vertical scans ----
        x_tb = tf.transpose(x, perm=[0, 2, 1, 3])   # (B, W, H, C)
        x_tb = tf.reshape(x_tb, (B * W, H, C))

        v_tb = self.scan_sequence(x_tb, reverse=False)
        v_tb = tf.reshape(v_tb, (B, W, H, C))
        v_tb = tf.transpose(v_tb, perm=[0, 2, 1, 3])

        v_bt = self.scan_sequence(x_tb, reverse=True)
        v_bt = tf.reshape(v_bt, (B, W, H, C))
        v_bt = tf.transpose(v_bt, perm=[0, 2, 1, 3])

        # ---- Directional fusion ----
        out = (h_lr + h_rl + v_tb + v_bt) / 4.0
        return out

  # =========================
# BLOCK 12.3: Context-Adaptive State Reset (CASR)
# =========================

class BoundaryAwareStateScan(layers.Layer):
    """
    Directional spatial state scan with boundary-aware adaptive reset.
    Prevents dominant-class memory collapse.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # State candidate projection
        self.state_proj = layers.Dense(channels)

        # Temporal/state gate (like Mamba but spatial)
        self.state_gate = layers.Dense(channels, activation="sigmoid")

        # Boundary confidence estimator
        self.boundary_proj = layers.Conv2D(
            1, kernel_size=3, padding="same", activation="sigmoid"
        )

    def scan_sequence(self, x, boundary, reverse=False):
        """
        x: (B, T, C)
        boundary: (B, T, 1)
        """
        if reverse:
            x = tf.reverse(x, axis=[1])
            boundary = tf.reverse(boundary, axis=[1])

        states = []
        s = tf.zeros_like(x[:, 0, :])

        for t in range(x.shape[1]):
            xt = x[:, t, :]
            bt = boundary[:, t, :]  # boundary confidence

            gate = self.state_gate(xt)
            candidate = self.state_proj(xt)

            # Boundary-aware reset
            s = (1.0 - bt) * (gate * s) + bt * candidate
            states.append(s)

        states = tf.stack(states, axis=1)

        if reverse:
            states = tf.reverse(states, axis=[1])

        return states

    def call(self, x):
        """
        x: (B, H, W, C)
        """
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        # ---- Boundary map ----
        boundary_map = self.boundary_proj(x)  # (B, H, W, 1)

        # ---- Horizontal scan ----
        x_lr = tf.reshape(x, (B * H, W, C))
        b_lr = tf.reshape(boundary_map, (B * H, W, 1))

        h_lr = self.scan_sequence(x_lr, b_lr, reverse=False)
        h_rl = self.scan_sequence(x_lr, b_lr, reverse=True)

        h_lr = tf.reshape(h_lr, (B, H, W, C))
        h_rl = tf.reshape(h_rl, (B, H, W, C))

        # ---- Vertical scan ----
        x_tb = tf.transpose(x, perm=[0, 2, 1, 3])  # (B, W, H, C)
        b_tb = tf.transpose(boundary_map, perm=[0, 2, 1, 3])

        x_tb = tf.reshape(x_tb, (B * W, H, C))
        b_tb = tf.reshape(b_tb, (B * W, H, 1))

        v_tb = self.scan_sequence(x_tb, b_tb, reverse=False)
        v_bt = self.scan_sequence(x_tb, b_tb, reverse=True)

        v_tb = tf.reshape(v_tb, (B, W, H, C))
        v_bt = tf.reshape(v_bt, (B, W, H, C))

        v_tb = tf.transpose(v_tb, perm=[0, 2, 1, 3])
        v_bt = tf.transpose(v_bt, perm=[0, 2, 1, 3])

        # ---- Directional fusion ----
        out = (h_lr + h_rl + v_tb + v_bt) / 4.0
        return out

  # =========================
# BLOCK 12.4: Dual-Scale State Interaction (DSSI)
# =========================

class DualScaleStateInteraction(layers.Layer):
    """
    Maintains micro- and macro-scale spatial states and allows
    bidirectional state-to-state modulation.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Micro-scale state scanner (high resolution)
        self.micro_state = BoundaryAwareStateScan(channels)

        # Macro-scale state scanner (lower resolution)
        self.macro_state = BoundaryAwareStateScan(channels)

        # State summarization
        self.micro_to_macro = layers.Dense(channels, activation="sigmoid")
        self.macro_to_micro = layers.Dense(channels, activation="sigmoid")

    def call(self, x):
        """
        x: (B, H, W, C)
        """

        # ---- Micro state (fine detail) ----
        micro = self.micro_state(x)  # (B, H, W, C)

        # ---- Macro state (coarse context) ----
        x_down = tf.nn.avg_pool2d(
            x, ksize=2, strides=2, padding="SAME"
        )  # (B, H/2, W/2, C)

        macro = self.macro_state(x_down)  # (B, H/2, W/2, C)

        # ---- Summarize states ----
        macro_summary = tf.reduce_mean(macro, axis=[1, 2])  # (B, C)
        micro_summary = tf.reduce_mean(micro, axis=[1, 2])  # (B, C)

        # ---- Cross-scale modulation ----
        micro_gate = self.macro_to_micro(macro_summary)
        macro_gate = self.micro_to_macro(micro_summary)

        micro_gate = tf.reshape(micro_gate, (-1, 1, 1, self.channels))
        macro_gate = tf.reshape(macro_gate, (-1, 1, 1, self.channels))

        micro = micro * micro_gate
        macro = macro * macro_gate

        # ---- Upsample macro back to micro resolution ----
        macro_up = tf.image.resize(
            macro,
            size=tf.shape(micro)[1:3],
            method="bilinear"
        )

        # ---- Final fusion (state-level, not feature concat) ----
        out = micro + macro_up
        return out

  # =========================
# BLOCK 12.5 (FIXED): State-Aware Decoder
# =========================

def state_aware_decoder(x, num_classes):
    """
    Decodes spatial state representations into segmentation logits
    at correct output resolution (256×256).
    """

    # ---- Boundary refinement ----
    boundary_refine = layers.Conv2D(
        1, kernel_size=3, padding="same", activation="sigmoid"
    )(x)

    # ---- State refinement ----
    x = layers.Conv2D(128, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # ---- Boundary-guided sharpening ----
    x = x * (1.0 + boundary_refine)

    # ---- Single upsampling (128 → 256) ----
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)

    # ---- Final projection ----
    outputs = layers.Conv2D(
        num_classes, kernel_size=1, activation="softmax"
    )(x)

    return outputs

# =========================
# BLOCK 12.6: InformalMamba-Net (Full Model)
# =========================

def InformalMambaNet(input_shape=(256, 256, 3), num_classes=7):
    inputs = layers.Input(shape=input_shape)

    # ---- CNN Stem ----
    x = informal_cnn_stem(inputs)        # (B, 128, 128, 64)

    # ---- Dual-Scale State Core ----
    x = DualScaleStateInteraction(64)(x) # (B, 128, 128, 64)

    # ---- State-Aware Decoder ----
    outputs = state_aware_decoder(x, num_classes)

    model = Model(inputs, outputs, name="InformalMamba-Net")
    return model


# ---- Build model ----
informal_mamba_model = InformalMambaNet(
    input_shape=(256, 256, 3),
    num_classes=num_classes
)

informal_mamba_model.summary()

# =========================
# BLOCK 13.1: Compile InformalMamba-Net
# =========================

informal_mamba_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=combined_dice_focal_loss,
    metrics=[dice_coefficient, mean_iou]
)

# =========================
# BLOCK 13.2: Train InformalMamba-Net
# =========================

history_mamba = informal_mamba_model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)

# =========================
# BLOCK 13.4: InformalMamba-Net v2 (Stacked State)
# =========================

def InformalMambaNet_v2(input_shape=(256, 256, 3), num_classes=7):
    inputs = layers.Input(shape=input_shape)

    # ---- Wider CNN Stem ----
    x = layers.Conv2D(48, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(96, 3, strides=2, padding="same", activation="relu")(x)  # 128×128

    # ---- Stacked Dual-Scale State Blocks ----
    x = DualScaleStateInteraction(96)(x)
    x = DualScaleStateInteraction(96)(x)

    # ---- State-Aware Decoder (correct resolution) ----
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)  # 256×256
    x = layers.Conv2D(96, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)

    outputs = layers.Conv2D(
        num_classes, kernel_size=1, activation="softmax"
    )(x)

    return Model(inputs, outputs, name="InformalMamba-Net-v2")


# ---- Build model ----
informal_mamba_v2 = InformalMambaNet_v2(
    input_shape=(256, 256, 3),
    num_classes=num_classes
)

informal_mamba_v2.summary()

# =========================
# BLOCK 13.5: Compile v2
# =========================

informal_mamba_v2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=combined_dice_focal_loss,
    metrics=[dice_coefficient, mean_iou]
)

# =========================
# BLOCK 13.7: Train InformalMamba-Net v2
# =========================

history_mamba_v2 = informal_mamba_v2.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)

# =========================
# BLOCK 14.1: Semantic Bottleneck Encoder (SBE)
# =========================

def semantic_bottleneck_encoder(inputs):
    """
    Purpose:
    - Extract high-level semantic features
    - Suppress raw texture noise
    - Reduce spatial resolution BEFORE state reasoning
    """

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)  # 128x128
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(192, 3, strides=2, padding="same", activation="relu")(x)  # 64x64
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)  # semantic-rich
    x = layers.BatchNormalization()(x)

    return x  # (B, 64, 64, 256)


# =========================
# BLOCK 14.2: Semantic-State Core (Decoupled)
# =========================

def semantic_state_core(x):
    """
    Apply InformalMamba reasoning ONLY on semantic features
    """

    x = layers.Conv2D(128, 1, padding="same", activation="relu")(x)

    # ---- Stacked State Blocks ----
    x = DualScaleStateInteraction(128)(x)
    x = DualScaleStateInteraction(128)(x)

    return x  # (B, 64, 64, 128)

# =========================
# BLOCK 14.3: Semantic Decoder
# =========================

def semantic_decoder(x, num_classes):
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)  # 128x128
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)

    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)  # 256x256
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)

    outputs = layers.Conv2D(num_classes, 1, activation="softmax")(x)
    return outputs

# =========================
# BLOCK 14.4: InformalMamba-Net v3 (SSDSR-Net)
# =========================

def InformalMambaNet_v3(input_shape=(256, 256, 3), num_classes=7):
    inputs = layers.Input(shape=input_shape)

    # ---- Semantic Encoder ----
    semantic_feats = semantic_bottleneck_encoder(inputs)

    # ---- Semantic-State Reasoning ----
    state_feats = semantic_state_core(semantic_feats)

    # ---- Decoder ----
    outputs = semantic_decoder(state_feats, num_classes)

    return Model(inputs, outputs, name="InformalMamba-Net-v3")


# ---- Build model ----
informal_mamba_v3 = InformalMambaNet_v3(
    input_shape=(256, 256, 3),
    num_classes=num_classes
)

informal_mamba_v3.summary()

# =========================
# BLOCK 14.5: Compile v3
# =========================

informal_mamba_v3.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=combined_dice_focal_loss,
    metrics=[dice_coefficient, mean_iou]
)

history_mamba_v3 = informal_mamba_v3.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,  
    validation_steps=val_steps,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)

# =========================
# BLOCK 15.1: Test Evaluation (Mean IoU Focus)
# =========================

def evaluate_model_iou(model, test_gen, steps, num_classes):
    ious, dices, precisions, recalls = [], [], [], []

    for _ in range(steps):
        x_batch, y_true = next(test_gen)
        y_pred = model.predict(x_batch, verbose=0)

        y_true_cls = np.argmax(y_true, axis=-1)
        y_pred_cls = np.argmax(y_pred, axis=-1)

        for i in range(y_true.shape[0]):
            yt = y_true_cls[i].flatten()
            yp = y_pred_cls[i].flatten()

            ious.append(jaccard_score(yt, yp, average="macro"))
            dices.append(f1_score(yt, yp, average="macro"))
            precisions.append(precision_score(yt, yp, average="macro", zero_division=0))
            recalls.append(recall_score(yt, yp, average="macro", zero_division=0))

    return {
        "Mean IoU": np.mean(ious),
        "Mean Dice": np.mean(dices),
        "Mean Precision": np.mean(precisions),
        "Mean Recall": np.mean(recalls),
    }


# =========================
# BLOCK 15.2: Test Evaluation
# =========================

test_steps = len(os.listdir(test_images)) // BATCH_SIZE

results_v1 = evaluate_model_iou(informal_mamba_model, test_gen, test_steps, num_classes)
results_v2 = evaluate_model_iou(informal_mamba_v2, test_gen, test_steps, num_classes)
results_v3 = evaluate_model_iou(informal_mamba_v3, test_gen, test_steps, num_classes)

print("InformalMamba v1:", results_v1)
print("InformalMamba v2:", results_v2)
print("InformalMamba v3 (SSDSR):", results_v3)

# =========================
# FIX: Safe onehot_to_rgb
# =========================

def onehot_to_rgb(onehot, colormap=id2code):
    """
    Convert one-hot mask to RGB using dataset colormap
    """
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros(onehot.shape[:2] + (3,), dtype=np.uint8)

    for class_id, color in colormap.items():
        output[single_layer == class_id] = color

    return output


# =========================
# BLOCK 15.3: Qualitative Visualization
# =========================

def visualize_predictions(models, model_names, image_dir, mask_dir, num_samples=3):
    images = sorted(os.listdir(image_dir))[:num_samples]
    
    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name.replace(".tif", ".png"))

        img = cv2.imread(img_path)
        img = cv2.resize(img, IMG_SIZE) / 255.0

        gt_mask = cv2.imread(mask_path)
        gt_mask = cv2.resize(gt_mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

        plt.figure(figsize=(15, 4))
        plt.subplot(1, len(models) + 2, 1)
        plt.imshow(img[..., ::-1])
        plt.title("Input")
        plt.axis("off")

        plt.subplot(1, len(models) + 2, 2)
        plt.imshow(gt_mask[..., ::-1])
        plt.title("Ground Truth")
        plt.axis("off")

        for i, model in enumerate(models):
            pred = model.predict(img[None, ...], verbose=0)
            pred_rgb = onehot_to_rgb(pred[0])

            plt.subplot(1, len(models) + 2, i + 3)
            plt.imshow(pred_rgb)
            plt.title(model_names[i])
            plt.axis("off")

        plt.show()

  # =========================
# BLOCK 15.4: Visual Comparison
# =========================

visualize_predictions(
    models=[informal_mamba_model, informal_mamba_v2, informal_mamba_v3],
    model_names=["Mamba v1", "Mamba v2", "Mamba v3 (SSDSR)"],
    image_dir=test_images,
    mask_dir=test_masks,
    num_samples=3
)

# =========================
# BLOCK 15.6: Visualize More Random Test Samples
# =========================

import random

def visualize_random_predictions(models, model_names, image_dir, mask_dir, num_samples=6):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".tif")])
    selected_images = random.sample(image_files, num_samples)

    for img_name in selected_images:
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name.replace(".tif", ".png"))

        img = cv2.imread(img_path)
        img = cv2.resize(img, IMG_SIZE) / 255.0

        gt_mask = cv2.imread(mask_path)
        gt_mask = cv2.resize(gt_mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

        plt.figure(figsize=(4 * (len(models) + 2), 4))

        plt.subplot(1, len(models) + 2, 1)
        plt.imshow(img[..., ::-1])
        plt.title("Input")
        plt.axis("off")

        plt.subplot(1, len(models) + 2, 2)
        plt.imshow(gt_mask[..., ::-1])
        plt.title("Ground Truth")
        plt.axis("off")

        for i, model in enumerate(models):
            pred = model.predict(img[None, ...], verbose=0)
            pred_rgb = onehot_to_rgb(pred[0])

            plt.subplot(1, len(models) + 2, i + 3)
            plt.imshow(pred_rgb)
            plt.title(model_names[i])
            plt.axis("off")

        plt.show()


visualize_random_predictions(
    models=[informal_mamba_model, informal_mamba_v2, informal_mamba_v3],
    model_names=["Mamba v1", "Mamba v2", "Mamba v3 (SSDSR)"],
    image_dir=test_images,
    mask_dir=test_masks,
    num_samples=6   # increase to 8–10 if you want
)

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def flatten_predictions(model, images, masks, num_classes, batch_size=4):
    """
    Returns flattened y_true, y_pred for metric computation
    """
    # Predict
    preds = model.predict(images, batch_size=batch_size, verbose=1)
    y_pred = np.argmax(preds, axis=-1)

    # Ground truth
    if masks.ndim == 4:  # one-hot
        y_true = np.argmax(masks, axis=-1)
    else:
        y_true = masks

    # Flatten
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    return y_true, y_pred

def compute_metrics(y_true, y_pred, num_classes):
    metrics = {}

    metrics["Accuracy"] = accuracy_score(y_true, y_pred)

    metrics["Precision_macro"] = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    metrics["Recall_macro"] = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    metrics["F1_macro"] = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Mean IoU
    ious = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else 0
        ious.append(iou)

    metrics["Mean_IoU"] = np.mean(ious)
    metrics["Per_Class_IoU"] = ious

    return metrics


models = {
    "DeepLabV3+": deeplab_model,
    "SegFormer": segformer_model,
    "Mamba v1": informal_mamba_model,
    "Mamba v2": informal_mamba_v2,
    "Mamba v3 (SSDSR)": informal_mamba_v3
}

all_results = {}

for name, model in models.items():
    print(f"\n===== Evaluating {name} =====")

    y_true, y_pred = flatten_predictions(
        model,
        test_images,
        test_masks,
        num_classes=num_classes
    )

    metrics = compute_metrics(y_true, y_pred, num_classes)
    cm = compute_confusion(y_true, y_pred, num_classes)

    all_results[name] = {
        "metrics": metrics,
        "confusion_matrix": cm
    }

    # Print metrics
    print(f"Accuracy        : {metrics['Accuracy']:.4f}")
    print(f"Precision (mac) : {metrics['Precision_macro']:.4f}")
    print(f"Recall (mac)    : {metrics['Recall_macro']:.4f}")
    print(f"F1-score (mac)  : {metrics['F1_macro']:.4f}")
    print(f"Mean IoU        : {metrics['Mean_IoU']:.4f}")
