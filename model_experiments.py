"""
Image-based Intrusion Detection System using CNN
================================================

This script implements a Convolutional Neural Network for intrusion detection
using image-based representation of network traffic data (NSL-KDD dataset).

Requirements:
- tensorflow>=2.8.0
- tensorflow-addons
- wandb
- sklearn
- seaborn
- matplotlib
- pandas
- numpy
- nvidia-ml-py3 (for GPU monitoring)

Author: Doo-Seop Choi
Date: 2025.05.30
License: MIT
"""

import pathlib
import wandb
import tqdm
import datetime
import re
import subprocess
import sys
import time
import os

import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import numpy as np

from tensorflow import keras
from wandb.keras import WandbCallback
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from datetime import timedelta

# Set random seeds for reproducibility
RANDOM_SEED = 123
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Configuration parameters
CONFIG = {
    'learning_rate': 0.01,
    'epochs': 100,
    'batch_size': 128,
    'img_height': 6,
    'img_width': 6,
    'experiment_count': 1,
    'validation_split': 0.2,
    'patience': 10,  # Early stopping patience
    'project_name': 'nsl_kdd',
    'entity': 'your_wandb_entity'  # Replace with your wandb entity
}


def check_dataset_paths():
    """
    Check if dataset paths exist and are accessible.

    Returns:
        tuple: (train_dir, test_dir) if paths exist, raises FileNotFoundError otherwise
    """
    # Update these paths according to your dataset location
    train_dir = pathlib.Path('./dataset/img_data/feature36_img/train/')
    test_dir = pathlib.Path('./dataset/img_data/feature36_img/val/')

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    print(f"✓ Training directory found: {train_dir}")
    print(f"✓ Test directory found: {test_dir}")

    return train_dir, test_dir


def create_learning_rate_scheduler():
    """
    Create learning rate scheduler that reduces learning rate after 10 epochs.

    Returns:
        tf.keras.callbacks.LearningRateScheduler: Learning rate scheduler callback
    """

    def scheduler(epoch, lr):
        """
        Learning rate scheduling function.

        Args:
            epoch (int): Current epoch number
            lr (float): Current learning rate

        Returns:
            float: New learning rate
        """
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    return tf.keras.callbacks.LearningRateScheduler(scheduler)


def load_datasets(train_dir, test_dir, config):
    """
    Load and prepare training, validation, and test datasets.

    Args:
        train_dir (pathlib.Path): Path to training data directory
        test_dir (pathlib.Path): Path to test data directory
        config (dict): Configuration parameters

    Returns:
        tuple: (train_ds, val_ds, test_ds, class_names)
    """
    print("Loading datasets...")

    # Load training dataset with validation split
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=train_dir,
        labels="inferred",
        validation_split=config['validation_split'],
        subset="training",
        seed=RANDOM_SEED,
        color_mode='grayscale',
        image_size=(config['img_height'], config['img_width']),
        batch_size=config['batch_size'],
    )

    # Load validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=train_dir,
        labels="inferred",
        validation_split=config['validation_split'],
        subset="validation",
        seed=RANDOM_SEED,
        color_mode='grayscale',
        image_size=(config['img_height'], config['img_width']),
        batch_size=config['batch_size'],
    )

    # Load test dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=test_dir,
        color_mode='grayscale',
        seed=RANDOM_SEED,
        image_size=(config['img_height'], config['img_width']),
        batch_size=config['batch_size'],
    )

    class_names = train_ds.class_names
    print(f"✓ Class names: {class_names}")
    print(f"✓ Number of classes: {len(class_names)}")

    # Optimize datasets for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def create_cnn_model(config):
    """
    Create CNN model architecture for binary classification.

    Args:
        config (dict): Configuration parameters

    Returns:
        tf.keras.Sequential: Compiled CNN model
    """
    print("Creating CNN model...")

    # Determine number of classes (1 for binary classification with sigmoid)
    num_classes = 1

    model = tf.keras.Sequential([
        # Input preprocessing layer
        tf.keras.layers.Rescaling(1. / 255, input_shape=(config['img_height'], config['img_width'], 1)),

        # First convolutional block
        tf.keras.layers.Conv2D(32, 3, padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.95),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        # Second convolutional block
        tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.95),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        # Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128),
        tf.keras.layers.BatchNormalization(momentum=0.95),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(num_classes),
        tf.keras.layers.Activation('sigmoid')  # Binary classification
    ])

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss="binary_crossentropy",
        metrics=['accuracy']
    )

    print("Model created and compiled successfully")
    model.summary()

    return model


def get_gpu_memory_usage():
    """
    Get current GPU memory usage statistics.

    Returns:
        float: Average GPU memory usage percentage across all GPUs
    """
    try:
        # Execute nvidia-smi command to get GPU memory info
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True,
            check=True
        )

        total_memory = 0
        memory_usage = 0
        num_gpus = 0

        # Parse output
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                gpu_stats = line.strip().split(',')
                memory_used = float(gpu_stats[0])
                memory_total = float(gpu_stats[1])
                memory_usage += memory_used
                total_memory += memory_total
                num_gpus += 1

        if num_gpus > 0:
            avg_memory_usage = (memory_usage / total_memory) * 100 / num_gpus
            return avg_memory_usage
        else:
            return 0.0

    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        print("Warning: Could not retrieve GPU memory usage")
        return 0.0


def plot_training_history(history, exp_iter):
    """
    Plot and save training history (accuracy and loss).

    Args:
        history: Training history object from model.fit()
        exp_iter (int): Experiment iteration number
    """
    # Create output directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)

    # Plot accuracy
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'plots/training_history_exp_{exp_iter}.png', dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_model(model, test_ds, class_names, exp_iter):
    """
    Evaluate model performance and generate metrics.

    Args:
        model: Trained Keras model
        test_ds: Test dataset
        class_names (list): List of class names
        exp_iter (int): Experiment iteration number

    Returns:
        dict: Evaluation metrics
    """
    print("Evaluating model on test dataset...")

    # Measure test time
    start_test_time = time.time()
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    end_test_time = time.time()
    test_duration = end_test_time - start_test_time

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Duration: {test_duration:.2f} seconds")

    # Generate predictions
    print("Generating predictions...")
    y_pred = []
    y_true = []
    y_prob = []

    for image_batch, label_batch in test_ds:
        probs = model.predict(image_batch, verbose=0)
        predictions = tf.where(probs > 0.5, 1, 0)
        predictions = tf.squeeze(predictions).numpy()

        # Handle single prediction case
        if predictions.ndim == 0:
            predictions = [predictions.item()]
        else:
            predictions = predictions.tolist()

        y_pred.extend(predictions)
        y_true.extend(label_batch.numpy().tolist())
        y_prob.extend(probs.flatten().tolist())

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    heatmap = sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    # Save confusion matrix
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/confusion_matrix_exp_{exp_iter}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/roc_curve_exp_{exp_iter}.png', dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'test_accuracy': test_accuracy,
        'test_duration': test_duration,
        'auc_score': auc_score,
        'y_true': y_true,
        'y_pred': y_pred,
        'confusion_matrix': cm
    }


def main():
    """
    Main function to run the complete training and evaluation pipeline.
    """
    print("=" * 60)
    print("Image-based Intrusion Detection System using CNN")
    print("=" * 60)

    try:
        # Check dataset paths
        train_dir, test_dir = check_dataset_paths()

        # Run experiments
        for exp_iter in range(CONFIG['experiment_count']):
            print(f"\n--- Experiment {exp_iter + 1}/{CONFIG['experiment_count']} ---")

            # Initialize Weights & Biases
            run = wandb.init(
                project=CONFIG['project_name'],
                entity=CONFIG['entity'],
                config=CONFIG,
                reinit=True
            )

            try:
                # Load datasets
                train_ds, val_ds, test_ds, class_names = load_datasets(train_dir, test_dir, CONFIG)

                # Create model
                model = create_cnn_model(CONFIG)

                # Setup callbacks
                callbacks = [
                    WandbCallback(),
                    tfa.callbacks.TQDMProgressBar(),
                    create_learning_rate_scheduler(),
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=CONFIG['patience'],
                        restore_best_weights=True,
                        verbose=1
                    )
                ]

                # Train model
                print("\nStarting model training...")
                start_time = datetime.datetime.now()

                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=CONFIG['epochs'],
                    callbacks=callbacks,
                    verbose=1
                )

                end_time = datetime.datetime.now()
                training_duration = end_time - start_time

                print(f"✓ Training completed in {training_duration}")

                # Log training metrics
                wandb.log({
                    'total_training_time_seconds': training_duration.total_seconds(),
                    'total_training_time_minutes': training_duration.total_seconds() / 60.0
                })

                # Log GPU usage
                gpu_usage = get_gpu_memory_usage()
                if gpu_usage > 0:
                    wandb.log({'avg_gpu_memory_usage_percent': gpu_usage})

                # Plot training history
                plot_training_history(history, exp_iter)

                # Evaluate model
                metrics = evaluate_model(model, test_ds, class_names, exp_iter)

                # Log evaluation metrics
                wandb.log({
                    'test_accuracy': metrics['test_accuracy'],
                    'test_duration_seconds': metrics['test_duration'],
                    'test_duration_minutes': metrics['test_duration'] / 60.0,
                    'auc_score': metrics['auc_score']
                })

                # Log confusion matrix to wandb
                wandb.log({
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        preds=metrics['y_pred'],
                        y_true=metrics['y_true'],
                        class_names=class_names
                    )
                })

                print(f"✓ Experiment {exp_iter + 1} completed successfully")

            except Exception as e:
                print(f" Error in experiment {exp_iter + 1}: {str(e)}")
                raise

            finally:
                # Finish wandb run
                wandb.finish()

    except Exception as e:
        print(f" Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()