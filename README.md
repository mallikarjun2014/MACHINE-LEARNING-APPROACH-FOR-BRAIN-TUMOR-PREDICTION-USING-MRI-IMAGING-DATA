# MACHINE-LEARNING-APPROACH-FOR-BRAIN-TUMOR-PREDICTION-USING-MRI-IMAGING-DATA
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from tensorflow.keras import backend as K
np.random.seed(42)
tf.random.set_seed(42)
#Defining dataset path
dataset_path = '/content/drive/MyDrive/brain_tumor_dataset'
categories = ['no', 'yes']
#Image and training parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_CNN = 50
EPOCHS_RESNET = 30
EPOCHS_FINETUNE = 20
#DATA LOADING AND EXPLORATION
def explore_dataset(dataset_path):
    """Explore the dataset structure and count images"""
    print("\n" + "="*70)
    print("DATASET EXPLORATION")
    print("="*70)
    data_info = {}
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        if os.path.exists(category_path):
            images = [img for img in os.listdir(category_path)
                     if img.endswith(('.jpg', '.jpeg', '.png'))]
            data_info[category] = len(images)
            print(f"Category '{category}': {len(images)} images")
        else:
            print(f"Warning: Path not found - {category_path}")
            data_info[category] = 0
    total_images = sum(data_info.values())
    print(f"\nTotal images in dataset: {total_images}")
    if total_images > 0:
        for category, count in data_info.items():
            percentage = (count / total_images) * 100
            label = 'No Tumor' if category == 'no' else 'Tumor'
            print(f"  - {label}: {percentage:.2f}%")
    plt.figure(figsize=(10, 6))
    colors = ['#2ECC71', '#E74C3C']
    bars = plt.bar(data_info.keys(), data_info.values(), color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    plt.title('Distribution of Brain Tumor Dataset', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Category', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Images', fontsize=14, fontweight='bold')
    plt.xticks(['no', 'yes'], ['No Tumor', 'Tumor Detected'], fontsize=12)
    for bar, (k, v) in zip(bars, data_info.items()):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(v)}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()
    return data_info
data_info = explore_dataset(dataset_path)
total_images = sum(data_info.values())
#VISUALIZING SAMPLE IMAGES
def display_sample_images(dataset_path, samples_per_category=6):
    """Display sample images from each category"""
    print("\n" + "="*70)
    print("SAMPLE IMAGES VISUALIZATION")
    print("="*70)
    fig, axes = plt.subplots(2, samples_per_category, figsize=(18, 6))
    fig.suptitle('Sample MRI Brain Images from Dataset', fontsize=18, fontweight='bold', y=1.02)
    for idx, category in enumerate(categories):
        category_path = os.path.join(dataset_path, category)
        images = [img for img in os.listdir(category_path)
                 if img.endswith(('.jpg', '.jpeg', '.png'))][:samples_per_category]
        for i, img_name in enumerate(images):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[idx, i].imshow(img)
            axes[idx, i].axis('off')
            if i == 0:
                label = 'NO TUMOR' if category == 'no' else 'TUMOR DETECTED'
                color = 'green' if category == 'no' else 'red'
                axes[idx, i].text(-0.1, 0.5, label,
                                transform=axes[idx, i].transAxes,
                                fontsize=14, fontweight='bold',
                                color=color, rotation=90,
                                va='center', ha='right')

    plt.tight_layout()
    plt.show()
    print("Sample images displayed successfully!")
display_sample_images(dataset_path)
#DATA PREPROCESSING AND LOADING
def load_and_preprocess_data(dataset_path, img_size=IMG_SIZE):
    """Load and preprocess all images"""
    print("\n" + "="*70)
    print("DATA PREPROCESSING")
    print("="*70)
    X = []
    y = []
    print("Loading and preprocessing images...")
    for category_idx, category in enumerate(categories):
        category_path = os.path.join(dataset_path, category)
        images = [img for img in os.listdir(category_path)
                 if img.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Processing '{category}' category...")
        for img_name in images:
            try:
                img_path = os.path.join(category_path, img_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size))
                X.append(img)
                y.append(category_idx)
            except Exception as e:
                print(f"  ✗ Error loading {img_name}: {e}")
    X = np.array(X, dtype='float32')
    y = np.array(y)
    X = X / 255.0
    print(f"\nData loaded successfully!")
    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")
    print(f"  - Data type: {X.dtype}")
    print(f"  - Value range: [{X.min():.2f}, {X.max():.2f}]")
    return X, y
X, y = load_and_preprocess_data(dataset_path)
#TRAIN-VALIDATION-TEST SPLIT
print("\n" + "="*70)
print("DATASET SPLITTING")
print("="*70)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)
print(f"Training set:   {X_train.shape[0]} images ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Validation set: {X_val.shape[0]} images ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test set:       {X_test.shape[0]} images ({X_test.shape[0]/len(X)*100:.1f}%)")
print("\nClass Distribution:")
for name, y_set in [('Training', y_train), ('Validation', y_val), ('Test', y_test)]:
    unique, counts = np.unique(y_set, return_counts=True)
    print(f"  {name}:")
    for cls, count in zip(unique, counts):
        label = 'No Tumor' if cls == 0 else 'Tumor'
        print(f"    - {label}: {count} ({count/len(y_set)*100:.1f}%)")
split_data = {
    'Training\n(64%)': X_train.shape[0],
    'Validation\n(16%)': X_val.shape[0],
    'Test\n(20%)': X_test.shape[0]
}
plt.figure(figsize=(10, 6))
colors = ['#3498DB', '#9B59B6', '#E67E22']
bars = plt.bar(split_data.keys(), split_data.values(), color=colors, alpha=0.8, edgecolor='black', linewidth=2)
plt.title('Dataset Split Distribution', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Number of Images', fontsize=14, fontweight='bold')
plt.xlabel('Dataset Split', fontsize=14, fontweight='bold')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()
#DATA AUGMENTATION SETUP
print("\n" + "="*70)
print("DATA AUGMENTATION CONFIGURATION")
print("="*70)
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)
val_test_datagen = ImageDataGenerator()
print("Training Data Augmentation:")
print("  - Rotation Range: ±15°")
print("  - Width/Height Shift: ±10%")
print("  - Horizontal Flip: Yes")
print("  - Zoom Range: ±10%")
print("  - Shear Range: ±10%")
print("\nValidation/Test Data: No augmentation")
def show_augmented_images(X_sample, n_augment=6):
    """Display original and augmented images"""
    fig, axes = plt.subplots(1, n_augment + 1, figsize=(18, 3))
    fig.suptitle('Data Augmentation Examples', fontsize=16, fontweight='bold')
    axes[0].imshow(X_sample)
    axes[0].set_title('Original', fontsize=12, fontweight='bold', color='blue')
    axes[0].axis('off')
    axes[0].add_patch(plt.Rectangle((0, 0), X_sample.shape[1]-1, X_sample.shape[0]-1,
                                   fill=False, edgecolor='blue', linewidth=3))
    X_sample_input = X_sample.reshape((1,) + X_sample.shape)
    aug_iter = train_datagen.flow(X_sample_input, batch_size=1)
    for i in range(1, n_augment + 1):
        aug_img = next(aug_iter)[0]
        axes[i].imshow(aug_img)
        axes[i].set_title(f'Augmented {i}', fontsize=12, fontweight='bold')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
print("\nGenerating augmentation examples...")
sample_idx = np.random.randint(0, len(X_train))
show_augmented_images(X_train[sample_idx])
#CUSTOM CNN ARCHITECTURE
print("\n" + "="*70)
print("MODEL 1: CUSTOM CNN ARCHITECTURE")
print("="*70)
def build_custom_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """Build a custom CNN model as per methodology"""
    model = Sequential(name='Custom_CNN')

    # Block 1
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape, name='conv1_1'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), name='pool1'))
    model.add(Dropout(0.25, name='dropout1'))

    # Block 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), name='pool2'))
    model.add(Dropout(0.25, name='dropout2'))

    # Block 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'))
    model.add(MaxPooling2D((2, 2), name='pool3'))
    model.add(Dropout(0.25, name='dropout3'))

    # Block 4
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_2'))
    model.add(MaxPooling2D((2, 2), name='pool4'))
    model.add(Dropout(0.25, name='dropout4'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(512, activation='relu', name='fc1'))
    model.add(Dropout(0.5, name='dropout5'))
    model.add(Dense(256, activation='relu', name='fc2'))
    model.add(Dropout(0.5, name='dropout6'))
    model.add(Dense(1, activation='sigmoid', name='output'))
    return model
custom_cnn = build_custom_cnn()
custom_cnn.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print("Custom CNN Model Architecture:")
custom_cnn.summary()
total_params = custom_cnn.count_params()
print(f"\nTotal Parameters: {total_params:,}")
#TRAINING CUSTOM CNN MODEL
print("\n" + "="*70)
print("TRAINING CUSTOM CNN MODEL")
print("="*70)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)
print(f"Training Configuration:")
print(f"  - Epochs: {EPOCHS_CNN}")
print(f"  - Batch Size: {BATCH_SIZE}")
print(f"  - Optimizer: Adam (lr=0.0001)")
print(f"  - Loss Function: Binary Crossentropy")
print(f"  - Early Stopping: patience=10")
print(f"  - Learning Rate Reduction: factor=0.5, patience=5")
print("\nStarting training...\n")
history_cnn = custom_cnn.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=EPOCHS_CNN,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
print("\nCustom CNN training completed!")
#RESNET50 WITH TRANSFER LEARNING
print("\n" + "="*70)
print("MODEL 2: RESNET50 WITH TRANSFER LEARNING")
print("="*70)
def build_resnet50_model(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """Build ResNet50 model with transfer learning"""
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False
    model = Sequential(name='ResNet50_Transfer')
    model.add(base_model)
    model.add(GlobalAveragePooling2D(name='global_avg_pool'))
    model.add(Dense(512, activation='relu', name='fc1'))
    model.add(Dropout(0.5, name='dropout1'))
    model.add(Dense(256, activation='relu', name='fc2'))
    model.add(Dropout(0.5, name='dropout2'))
    model.add(Dense(1, activation='sigmoid', name='output'))
    return model
print("Loading pre-trained ResNet50 (ImageNet weights)...")
resnet_model = build_resnet50_model()
resnet_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print("ResNet50 Model Architecture:")
resnet_model.summary()
total_params = resnet_model.count_params()
trainable_params = sum([tf.keras.backend.count_params(w) for w in resnet_model.trainable_weights])
print(f"\nTotal Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
#TRAINING RESNET50 MODEL
print("\n" + "="*70)
print("TRAINING RESNET50 MODEL - PHASE 1 (Feature Extraction)")
print("="*70)
print(f"Training Configuration:")
print(f"  - Epochs: {EPOCHS_RESNET}")
print(f"  - Batch Size: {BATCH_SIZE}")
print(f"  - Base Model: Frozen (Feature Extraction)")
print(f"  - Optimizer: Adam (lr=0.0001)")
print("\nStarting training...\n")
history_resnet = resnet_model.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=EPOCHS_RESNET,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
#FINE-TUNING RESNET50
print("\n" + "="*70)
print("TRAINING RESNET50 MODEL - PHASE 2 (Fine-Tuning)")
print("="*70)
base_model = resnet_model.layers[0]
base_model.trainable = True
print("Unfreezing last 20 layers of ResNet50 base model...")
for layer in base_model.layers[:-20]:
    layer.trainable = False
trainable_params = sum([tf.keras.backend.count_params(w) for w in resnet_model.trainable_weights])
print(f"Trainable Parameters after unfreezing: {trainable_params:,}")
resnet_model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print(f"\nFine-tuning Configuration:")
print(f"  - Epochs: {EPOCHS_FINETUNE}")
print(f"  - Batch Size: {BATCH_SIZE}")
print(f"  - Unfrozen Layers: Last 20 layers")
print(f"  - Optimizer: Adam (lr=0.00001)")
print("\nContinuing training...\n")
history_resnet_fine = resnet_model.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=EPOCHS_FINETUNE,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
print("\nResNet50 Phase 2 (Fine-tuning) completed!")
history_resnet_combined = {
    'accuracy': history_resnet.history['accuracy'] + history_resnet_fine.history['accuracy'],
    'val_accuracy': history_resnet.history['val_accuracy'] + history_resnet_fine.history['val_accuracy'],
    'loss': history_resnet.history['loss'] + history_resnet_fine.history['loss'],
    'val_loss': history_resnet.history['val_loss'] + history_resnet_fine.history['val_loss']
}
#PLOTTING TRAINING HISTORY
def plot_training_history(history, model_name, phase=None):
    """Plot training and validation accuracy/loss"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    title = f'{model_name} - Training History'
    if phase:
        title += f' ({phase})'
    fig.suptitle(title, fontsize=18, fontweight='bold')
    axes[0].plot(history.history['accuracy'], label='Training Accuracy',
                linewidth=2.5, marker='o', markersize=4, color='#3498DB')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy',
                linewidth=2.5, marker='s', markersize=4, color='#E74C3C')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=11)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_ylim([0, 1])
    axes[1].plot(history.history['loss'], label='Training Loss',
                linewidth=2.5, marker='o', markersize=4, color='#3498DB')
    axes[1].plot(history.history['val_loss'], label='Validation Loss',
                linewidth=2.5, marker='s', markersize=4, color='#E74C3C')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=11)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()
print("\n" + "="*70)
print("TRAINING HISTORY VISUALIZATION")
print("="*70)
print("\n1. Custom CNN Model:")
plot_training_history(history_cnn, 'Custom CNN')
print("\n2. ResNet50 Model - Phase 1 (Feature Extraction):")
plot_training_history(history_resnet, 'ResNet50', 'Phase 1: Feature Extraction')
print("\n3. ResNet50 Model - Phase 2 (Fine-Tuning):")
plot_training_history(history_resnet_fine, 'ResNet50', 'Phase 2: Fine-Tuning')
print("\n4. ResNet50 Model - Complete Training:")
class CombinedHistory:
    def __init__(self, history_dict):
        self.history = history_dict
combined_hist = CombinedHistory(history_resnet_combined)
plot_training_history(combined_hist, 'ResNet50', 'Complete Training')
##MODEL EVALUATION FUNCTION
def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation with all metrics"""
    print("\n" + "="*70)
    print(f"{model_name.upper()} - EVALUATION ON TEST SET")
    print("="*70)
    print("\nGenerating predictions...")
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_test, y_pred_prob)
    print("\n" + "-"*70)
    print("PERFORMANCE METRICS:")
    print("-"*70)
    print(f"{'Metric':<20} {'Score':<15} {'Percentage':<15}")
    print("-"*70)
    print(f"{'Accuracy':<20} {accuracy:<15.4f} {accuracy*100:<15.2f}%")
    print(f"{'Precision':<20} {precision:<15.4f} {precision*100:<15.2f}%")
    print(f"{'Recall':<20} {recall:<15.4f} {recall*100:<15.2f}%")
    print(f"{'F1-Score':<20} {f1:<15.4f} {f1*100:<15.2f}%")
    print(f"{'AUC-ROC':<20} {auc_roc:<15.4f} {auc_roc*100:<15.2f}%")
    print("-"*70)
    print("\n" + "-"*70)
    print("DETAILED CLASSIFICATION REPORT:")
    print("-"*70)
    print(classification_report(y_test, y_pred,
                               target_names=['No Tumor (Class 0)', 'Tumor (Class 1)'],
                               digits=4))
    def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation with all metrics"""
    print("\n" + "="*70)
    print(f"{model_name.upper()} - EVALUATION ON TEST SET")
    print("="*70)
    print("\nGenerating predictions...")
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_test, y_pred_prob)
    print("\n" + "-"*70)
    print("PERFORMANCE METRICS:")
    print("-"*70)
    print(f"{'Metric':<20} {'Score':<15} {'Percentage':<15}")
    print("-"*70)
    print(f"{'Accuracy':<20} {accuracy:<15.4f} {accuracy*100:<15.2f}%")
    print(f"{'Precision':<20} {precision:<15.4f} {precision*100:<15.2f}%")
    print(f"{'Recall':<20} {recall:<15.4f} {recall*100:<15.2f}%")
    print(f"{'F1-Score':<20} {f1:<15.4f} {f1*100:<15.2f}%")
    print(f"{'AUC-ROC':<20} {auc_roc:<15.4f} {auc_roc*100:<15.2f}%")
    print("-"*70)
    print("\n" + "-"*70)
    print("DETAILED CLASSIFICATION REPORT:")
    print("-"*70)
    print(classification_report(y_test, y_pred,
                               target_names=['No Tumor (Class 0)', 'Tumor (Class 1)'],
                               digits=4))
    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{model_name} - Evaluation Visualizations', fontsize=18, fontweight='bold')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Tumor', 'Tumor'],
                yticklabels=['No Tumor', 'Tumor'],
                annot_kws={"size": 16, "weight": "bold"},
                linewidths=2, linecolor='black',
                ax=axes[0])
    axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_ylabel('Actual Label', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    tn, fp, fn, tp = cm.ravel()
    metrics_text = f'TN={tn}  FP={fp}\nFN={fn}  TP={tp}'
    axes[0].text(0.5, -0.15, metrics_text, ha='center', transform=axes[0].transAxes,
                fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    axes[1].plot(fpr, tpr, color='#E74C3C', linewidth=3, label=f'ROC Curve (AUC = {auc_roc:.4f})')
    axes[1].plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--', label='Random Classifier')
    axes[1].fill_between(fpr, tpr, alpha=0.2, color='#E74C3C')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    axes[1].set_title('ROC Curve', fontsize=14, fontweight='bold', pad=15)
    axes[1].legend(loc="lower right", fontsize=11)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob
    }
    return results
    #EVALUATION OF BOTH MODELS
    #Evaluation of Custom CNN
    print("\n" + "="*70)
print("MODEL EVALUATION PHASE")
print("="*70)
results_cnn = evaluate_model(custom_cnn, X_test, y_test, "Custom CNN")
#Evaluation of ResNet50
results_resnet = evaluate_model(resnet_model, X_test, y_test, "ResNet50 Transfer Learning")
#COMPARATIVE ANALYSIS
print("\n" + "="*70)
print("COMPARATIVE ANALYSIS: CUSTOM CNN vs RESNET50")
print("="*70)
comparison_df = pd.DataFrame({
    'Custom CNN': [
        results_cnn['accuracy'],
        results_cnn['precision'],
        results_cnn['recall'],
        results_cnn['f1_score'],
        results_cnn['auc_roc']
    ],
    'ResNet50': [
        results_resnet['accuracy'],
        results_resnet['precision'],
        results_resnet['recall'],
        results_resnet['f1_score'],
        results_resnet['auc_roc']
    ]
}, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'])
print("\nPerformance Comparison Table:")
print("-"*70)
print(comparison_df.round(4))
print("-"*70)
best_model_name = 'ResNet50' if results_resnet['accuracy'] > results_cnn['accuracy'] else 'Custom CNN'
print(f"\n✓ Best Performing Model: {best_model_name}")
print(f"  - Accuracy: {max(results_resnet['accuracy'], results_cnn['accuracy']):.4f}")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Model Performance Comparison', fontsize=18, fontweight='bold')
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
x = np.arange(len(metrics))
width = 0.35
bars1 = axes[0].bar(x - width/2, comparison_df['Custom CNN'], width,
                    label='Custom CNN', color='#3498DB', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = axes[0].bar(x + width/2, comparison_df['ResNet50'], width,
                    label='ResNet50', color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1.5)
axes[0].set_xlabel('Metrics', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Score', fontsize=13, fontweight='bold')
axes[0].set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold', pad=15)
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics, fontsize=11, rotation=15)
axes[0].legend(fontsize=12)
axes[0].grid(axis='y', alpha=0.3, linestyle='--')
axes[0].set_ylim([0, 1.1])
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]
cnn_values = comparison_df['Custom CNN'].tolist()
cnn_values += cnn_values[:1]
resnet_values = comparison_df['ResNet50'].tolist()
resnet_values += resnet_values[:1]
ax2 = plt.subplot(122, projection='polar')
ax2.plot(angles, cnn_values, 'o-', linewidth=2.5, label='Custom CNN', color='#3498DB')
ax2.fill(angles, cnn_values, alpha=0.25, color='#3498DB')
ax2.plot(angles, resnet_values, 's-', linewidth=2.5, label='ResNet50', color='#E74C3C')
ax2.fill(angles, resnet_values, alpha=0.25, color='#E74C3C')
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(metrics, fontsize=11)
ax2.set_ylim(0, 1)
ax2.set_title('Radar Chart Comparison', fontsize=14, fontweight='bold', pad=20)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
##PREDICTION VISUALIZATION
print("\n" + "="*70)
print("PREDICTION VISUALIZATION ON TEST SAMPLES")
print("="*70)
def visualize_predictions(model, X_test, y_test, model_name, n_samples=12):
    """Visualize model predictions on test samples"""
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    predictions = model.predict(X_test[indices], verbose=0)
    pred_classes = (predictions > 0.5).astype(int).flatten()
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle(f'{model_name} - Sample Predictions on Test Set', fontsize=18, fontweight='bold')
    for idx, ax in enumerate(axes.flat):
        if idx < len(indices):
            img = X_test[indices[idx]]
            true_label = y_test[indices[idx]]
            pred_label = pred_classes[idx]
            confidence = predictions[idx][0]
            ax.imshow(img)
            ax.axis('off')
            is_correct = (true_label == pred_label)
            border_color = 'green' if is_correct else 'red'
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(4)
            true_text = 'Tumor' if true_label == 1 else 'No Tumor'
            pred_text = 'Tumor' if pred_label == 1 else 'No Tumor'

            title = f'True: {true_text}\nPred: {pred_text}\nConf: {confidence:.2%}'
            title_color = 'green' if is_correct else 'red'
            ax.set_title(title, fontsize=11, fontweight='bold', color=title_color)
    plt.tight_layout()
    plt.show()
print("\n1. Custom CNN Predictions:")
visualize_predictions(custom_cnn, X_test, y_test, "Custom CNN")
print("\n2. ResNet50 Predictions:")
visualize_predictions(resnet_model, X_test, y_test, "ResNet50 Transfer Learning")
#ERROR ANALYSIS
print("\n" + "="*70)
print("ERROR ANALYSIS")
print("="*70)
def analyze_errors(model, X_test, y_test, model_name):
    """Analyze misclassified samples"""
    print(f"\n{model_name} - Error Analysis:")
    print("-"*70)
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    errors = np.where(y_pred != y_test)[0]
    false_positives = np.where((y_pred == 1) & (y_test == 0))[0]
    false_negatives = np.where((y_pred == 0) & (y_test == 1))[0]
    print(f"Total Errors: {len(errors)} ({len(errors)/len(y_test)*100:.2f}%)")
    print(f"False Positives: {len(false_positives)} ({len(false_positives)/len(y_test)*100:.2f}%)")
    print(f"False Negatives: {len(false_negatives)} ({len(false_negatives)/len(y_test)*100:.2f}%)")
    if len(errors) > 0:
        n_show = min(8, len(errors))
        error_indices = np.random.choice(errors, n_show, replace=False)
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        fig.suptitle(f'{model_name} - Misclassified Samples', fontsize=18, fontweight='bold')
        for idx, ax in enumerate(axes.flat):
            if idx < len(error_indices):
                err_idx = error_indices[idx]
                img = X_test[err_idx]
                true_label = y_test[err_idx]
                pred_label = y_pred[err_idx]
                confidence = y_pred_prob[err_idx][0]
                ax.imshow(img)
                ax.axis('off')
                true_text = 'Tumor' if true_label == 1 else 'No Tumor'
                pred_text = 'Tumor' if pred_label == 1 else 'No Tumor'
                title = f'True: {true_text}\nPredicted: {pred_text}\nConfidence: {confidence:.2%}'
                ax.set_title(title, fontsize=11, fontweight='bold', color='red')
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(4)
        plt.tight_layout()
        plt.show()
analyze_errors(custom_cnn, X_test, y_test, "Custom CNN")
analyze_errors(resnet_model, X_test, y_test, "ResNet50")
#CONFIDENCE DISTRIBUTION ANALYSIS
print("\n" + "="*70)
print("CONFIDENCE DISTRIBUTION ANALYSIS")
print("="*70)
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Prediction Confidence Distribution Analysis', fontsize=18, fontweight='bold')
y_pred_cnn = results_cnn['y_pred_prob'].flatten()
correct_cnn = (results_cnn['y_pred'] == y_test)

axes[0, 0].hist(y_pred_cnn[correct_cnn], bins=50, alpha=0.7, color='green',
                label='Correct', edgecolor='black')
axes[0, 0].hist(y_pred_cnn[~correct_cnn], bins=50, alpha=0.7, color='red',
                label='Incorrect', edgecolor='black')
axes[0, 0].axvline(x=0.5, color='blue', linestyle='--', linewidth=2, label='Threshold')
axes[0, 0].set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Custom CNN - Confidence Distribution', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)
y_pred_resnet = results_resnet['y_pred_prob'].flatten()
correct_resnet = (results_resnet['y_pred'] == y_test)

axes[0, 1].hist(y_pred_resnet[correct_resnet], bins=50, alpha=0.7, color='green',
                label='Correct', edgecolor='black')
axes[0, 1].hist(y_pred_resnet[~correct_resnet], bins=50, alpha=0.7, color='red',
                label='Incorrect', edgecolor='black')
axes[0, 1].axvline(x=0.5, color='blue', linestyle='--', linewidth=2, label='Threshold')
axes[0, 1].set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0, 1].set_title('ResNet50 - Confidence Distribution', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3)
data_cnn = [y_pred_cnn[y_test==0], y_pred_cnn[y_test==1]]
bp1 = axes[1, 0].boxplot(data_cnn, labels=['No Tumor', 'Tumor'],
                          patch_artist=True, widths=0.6)
for patch, color in zip(bp1['boxes'], ['#2ECC71', '#E74C3C']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1, 0].set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Custom CNN - Confidence by Class', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')
data_resnet = [y_pred_resnet[y_test==0], y_pred_resnet[y_test==1]]
bp2 = axes[1, 1].boxplot(data_resnet, labels=['No Tumor', 'Tumor'],
                          patch_artist=True, widths=0.6)
for patch, color in zip(bp2['boxes'], ['#2ECC71', '#E74C3C']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1, 1].set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
axes[1, 1].set_title('ResNet50 - Confidence by Class', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
#MODEL COMPARISON SUMMARY TABLE
print("\n" + "="*70)
print("COMPREHENSIVE MODEL COMPARISON SUMMARY")
print("="*70)
summary_data = {
    'Model': ['Custom CNN', 'ResNet50 Transfer Learning'],
    'Architecture': ['4 Conv Blocks + Dense', 'Pre-trained ResNet50 + Dense'],
    'Total Parameters': [
        f"{custom_cnn.count_params():,}",
        f"{resnet_model.count_params():,}"
    ],
    'Training Time': ['~Medium', '~Fast (Transfer Learning)'],
    'Accuracy': [
        f"{results_cnn['accuracy']:.4f} ({results_cnn['accuracy']*100:.2f}%)",
        f"{results_resnet['accuracy']:.4f} ({results_resnet['accuracy']*100:.2f}%)"
    ],
    'Precision': [
        f"{results_cnn['precision']:.4f}",
        f"{results_resnet['precision']:.4f}"
    ],
    'Recall': [
        f"{results_cnn['recall']:.4f}",
        f"{results_resnet['recall']:.4f}"
    ],
    'F1-Score': [
        f"{results_cnn['f1_score']:.4f}",
        f"{results_resnet['f1_score']:.4f}"
    ],
    'AUC-ROC': [
        f"{results_cnn['auc_roc']:.4f}",
        f"{results_resnet['auc_roc']:.4f}"
    ],
    'Advantage': [
        'Custom design, Interpretable',
        'Pre-trained features, Higher accuracy'
    ]
}
summary_df = pd.DataFrame(summary_data)
print("\n")
print(summary_df.to_string(index=False))
print("-"*70)
