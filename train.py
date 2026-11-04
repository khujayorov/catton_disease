"""Train and evaluate a CNN on the Cotton Disease directory dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a CNN for cotton disease image classification."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing train/, val/, and test/ (default: project directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for the model, metrics, and plots.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 1:
        parser.error("--epochs, --batch-size, and --image-size must be positive integers.")
    return args


def validate_dataset(data_dir: Path) -> list[str]:
    missing = [split for split in SPLITS if not (data_dir / split).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Dataset split folder(s) not found in {data_dir}: {', '.join(missing)}"
        )

    class_sets = {
        split: {
            item.name
            for item in (data_dir / split).iterdir()
            if item.is_dir()
        }
        for split in SPLITS
    }
    classes = sorted(class_sets["train"])
    if not classes:
        raise ValueError(f"No class folders found in {data_dir / 'train'}.")

    for split, split_classes in class_sets.items():
        if split_classes != set(classes):
            raise ValueError(
                f"Class folders in {data_dir / split} do not match train/: "
                f"expected {classes}, found {sorted(split_classes)}."
            )
        for class_name in classes:
            image_files = [
                path
                for path in (data_dir / split / class_name).rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if not image_files:
                raise ValueError(
                    f"No supported images found in {data_dir / split / class_name}."
                )
    return classes


def load_datasets(
    data_dir: Path, class_names: list[str], image_size: int, batch_size: int, seed: int
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    datasets = []
    for split in SPLITS:
        dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir / split,
            labels="inferred",
            label_mode="int",
            class_names=class_names,
            image_size=(image_size, image_size),
            batch_size=batch_size,
            shuffle=split == "train",
            seed=seed,
        )
        datasets.append(dataset.prefetch(tf.data.AUTOTUNE))
    return datasets[0], datasets[1], datasets[2]


def build_model(image_size: int, class_count: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    x = tf.keras.layers.RandomFlip("horizontal")(x)
    x = tf.keras.layers.RandomRotation(0.08)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)

    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def class_weights(data_dir: Path, class_names: list[str]) -> dict[int, float]:
    counts = np.array(
        [
            sum(
                path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                for path in (data_dir / "train" / class_name).rglob("*")
            )
            for class_name in class_names
        ]
    )
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_names)),
        y=np.repeat(np.arange(len(class_names)), counts),
    )
    return {index: float(weight) for index, weight in enumerate(weights)}


def save_training_plot(history: tf.keras.callbacks.History, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, metric, title in zip(
        axes, ("accuracy", "loss"), ("Accuracy", "Loss")
    ):
        axis.plot(history.history[metric], label="Train")
        axis.plot(history.history[f"val_{metric}"], label="Validation")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.legend()
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_confusion_matrix(
    labels: np.ndarray, predictions: np.ndarray, class_names: list[str], output_path: Path
) -> None:
    matrix = confusion_matrix(labels, predictions, labels=np.arange(len(class_names)))
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Test confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    class_names = validate_dataset(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)

    train_ds, val_ds, test_ds = load_datasets(
        data_dir, class_names, args.image_size, args.batch_size, args.seed
    )
    model = build_model(args.image_size, len(class_names))
    model_path = output_dir / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights(data_dir, class_names),
        callbacks=callbacks,
        shuffle=False,
    )
    save_training_plot(history, output_dir / "training_history.png")

    best_model = tf.keras.models.load_model(model_path)
    labels = np.concatenate([batch_labels.numpy() for _, batch_labels in test_ds])
    probabilities = best_model.predict(test_ds, verbose=1)
    predictions = np.argmax(probabilities, axis=1)
    report = classification_report(
        labels,
        predictions,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        output_dir / "classification_report.csv", index_label="class"
    )
    metrics = {
        "accuracy": float(report["accuracy"]),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "precision_weighted": float(report["weighted avg"]["precision"]),
        "recall_weighted": float(report["weighted avg"]["recall"]),
        "f1_weighted": float(report["weighted avg"]["f1-score"]),
        "class_names": class_names,
        "test_image_count": int(len(labels)),
        "best_validation_loss": float(min(history.history["val_loss"])),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "class_names.json").write_text(
        json.dumps(class_names, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_confusion_matrix(
        labels, predictions, class_names, output_dir / "confusion_matrix.png"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved model and evaluation results to: {output_dir}")


if __name__ == "__main__":
    main()
