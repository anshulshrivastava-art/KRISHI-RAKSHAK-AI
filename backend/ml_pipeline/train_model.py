"""
Krishi Rakshak AI - Model Training Pipeline
Team: Codex Verse
Model: MobileNetV2 Transfer Learning for Crop Disease Classification
Architecture: Frozen MobileNetV2 Backbone -> GlobalAveragePooling2D -> Dense Head -> Fine-Tuning
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def get_data_pipelines(batch_size: int = config.BATCH_SIZE):
    """
    Load dataset from train and val directories using tf.keras.utils.image_dataset_from_directory.
    Ensures zero data leakage between training and validation sets.
    """
    if not config.TRAIN_DIR.exists() or not any(config.TRAIN_DIR.iterdir()):
        raise FileNotFoundError(
            f"Training data not found at {config.TRAIN_DIR}. "
            "Please run 'python download_dataset.py' first!"
        )

    print(f"Loading datasets from {config.DATA_DIR}...")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        image_size=config.IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=42
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.VAL_DIR,
        image_size=config.IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False
    )

    class_names = train_ds.class_names
    print(f"Discovered {len(class_names)} classes: {class_names}")

    # Optimize data pipeline performance for CPU
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def build_crop_disease_model(num_classes: int) -> tf.keras.Model:
    """
    Build transfer learning model with MobileNetV2 base and custom classification head.
    Includes training-only data augmentation and normalization.
    """
    print("\nInitializing MobileNetV2 Transfer Learning Architecture...")
    
    # 1. Training Augmentation Layers
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.12),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

    # 2. Rescaling to [-1, 1] for MobileNetV2
    rescaling = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenetv2_rescaling")

    # 3. Base MobileNetV2 Model (Pretrained on ImageNet)
    base_model = MobileNetV2(
        input_shape=config.INPUT_SHAPE,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze backbone for initial head training

    # 4. Model Assembly
    inputs = tf.keras.Input(shape=config.INPUT_SHAPE, name="leaf_image_input")
    x = data_augmentation(inputs)
    x = rescaling(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm_1")(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="disease_prediction")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Krishi_Rakshak_MobileNetV2")
    return model, base_model


def train(epochs: int = config.EPOCHS, fine_tune: bool = True):
    print("=" * 70)
    print("[*] KRISHI RAKSHAK AI - MODEL TRAINING PIPELINE")
    print(f"Team: {config.TEAM_NAME}")
    print(f"Base Architecture: MobileNetV2 (ImageNet Weights)")
    print(f"Input Shape: {config.INPUT_SHAPE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print("=" * 70)

    train_ds, val_ds, class_names = get_data_pipelines()
    num_classes = len(class_names)

    # Save class label index mapping
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    class_indices = {i: name for i, name in enumerate(class_names)}
    with open(config.CLASS_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_indices, f, indent=2)
    print(f"Saved class label mapping to {config.CLASS_INDICES_PATH}")

    model, base_model = build_crop_disease_model(num_classes)
    
    # Compile with Adam optimizer and Categorical Crossentropy
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary(print_fn=lambda x: print(f"  {x}"))

    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    print(f"\nPhase 1: Training Classification Head ({epochs} epochs)...")
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stop, reduce_lr]
    )

    # Phase 2: Optional Fine-tuning of top layers
    if fine_tune:
        print("\nPhase 2: Fine-Tuning Top 20 Layers of MobileNetV2...")
        base_model.trainable = True
        # Freeze all layers except top 20
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        model.compile(
            optimizer=optimizers.Adam(learning_rate=config.FINE_TUNE_LEARNING_RATE),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        history_finetune = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.FINE_TUNE_EPOCHS,
            callbacks=[early_stop, reduce_lr]
        )

    # Save trained model in native Keras format
    print(f"\nSaving trained model to {config.MODEL_PATH}...")
    model.save(str(config.MODEL_PATH))
    try:
        model.save(str(config.MODEL_H5_PATH))
    except Exception:
        pass

    print("\n" + "=" * 70)
    print("[SUCCESS] MODEL TRAINING COMPLETE!")
    print(f"Artifact Saved: {config.MODEL_PATH}")
    print(f"Class Mapping: {config.CLASS_INDICES_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Krishi Rakshak MobileNetV2 model.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of head epochs")
    parser.add_argument("--no-fine-tune", action="store_true", help="Skip fine-tuning phase")
    args = parser.parse_args()

    train(epochs=args.epochs, fine_tune=not args.no_fine_tune)
