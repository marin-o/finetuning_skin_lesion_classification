import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

from itertools import cycle

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score


SEED = 42
NUM_CLASSES = 7

def load_data(train_metadata_path: str = 'data/ham10000/HAM10000_metadata.csv',
              test_metadata_path: str = 'data/ham10000/images/ISIC2018_Task3_Test_GroundTruth.csv',
              train_images_path: str = 'data/ham10000/images/',
              test_images_path: str = 'data/ham10000/images/test_images/',
              ) -> None:
    """
    Load and preprocess training and testing data for skin lesion classification.

    This function reads metadata and image paths for training and testing datasets,
    encodes the labels, and splits the training data into training and validation sets.
    It also visualizes the distribution of classes in the training and testing datasets.

    Parameters:
    - train_metadata_path (str): Path to the training metadata CSV file.
    - test_metadata_path (str): Path to the testing metadata CSV file.
    - train_images_path (str): Directory path where training images are stored.
    - test_images_path (str): Directory path where testing images are stored.

    Expected metadata columns:
    - image_id (str): Unique identifier for each image.
    - dx (str): Diagnosis label for each image.

    Returns:
    - train_df (pd.DataFrame): DataFrame containing the training data.
    - val_df (pd.DataFrame): DataFrame containing the validation data.
    - test_df (pd.DataFrame): DataFrame containing the testing data.

    Example usage:
    ```
        load_data(
            train_metadata_path='data/ham10000/HAM10000_metadata.csv',
            test_metadata_path='data/ham10000/images/ISIC2018_Task3_Test_GroundTruth.csv',
            train_images_path='data/ham10000/images/',
            test_images_path='data/ham10000/images/test_images/'
        )
    ```
    """
    df = pd.read_csv(train_metadata_path)

    df['image_path'] = train_images_path + df['image_id'] + '.jpg'

    df.head()

    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['dx'])

    sns.countplot(x='dx', data=df)
    plt.title('Train data')
    plt.show()

    test_df = pd.read_csv(test_metadata_path)
    test_df['image_path'] = test_images_path + test_df['image_id'] + '.jpg'
    test_df['label'] = label_encoder.transform(test_df['dx'])

    sns.countplot(x='dx', data=test_df)
    plt.title('Test data')
    plt.show()

    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=SEED)
    print(train_df.shape, val_df.shape, test_df.shape)
    val_df.reset_index(inplace=True, drop=True)
    return train_df, val_df, test_df


def show_image(image: Tensor, mean: float, std: float):
    mean = torch.tensor(mean, dtype=torch.float32)
    std = torch.tensor(std, dtype=torch.float32)
    image = image * std[:, None, None] + mean[:, None, None]
    image = image.permute(1, 2, 0)
    
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def train_model(
    model: nn.Module, 
    train_dataloader: DataLoader, 
    val_dataloader: DataLoader,  
    criterion, 
    optimizer, 
    scheduler,  
    device, 
    num_epochs=5, 
    log_interval=10
):
    model.to(device)

    for epoch in range(num_epochs):
        ### 🔹 TRAINING PHASE ###
        model.train()
        running_loss = 0.0
        epoch_loss = 0.0  
        total_batches = len(train_dataloader)

        print(f"🔹 Starting epoch {epoch+1}/{num_epochs}")

        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(0)  
            elif inputs.dim() == 2:
                inputs = inputs.unsqueeze(0).unsqueeze(0)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()

            if (i + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                print(f'[Train] Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{total_batches} - Loss: {avg_loss:.4f}')
                running_loss = 0.0

        avg_train_loss = epoch_loss / total_batches
        print(f'✅ Epoch {epoch+1} Finished. Train Loss: {avg_train_loss:.4f}')

        ### 🔹 VALIDATION PHASE ###
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                if inputs.dim() == 3:
                    inputs = inputs.unsqueeze(0)  
                elif inputs.dim() == 2:
                    inputs = inputs.unsqueeze(0).unsqueeze(0)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)  
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = correct / total * 100

        print(f'📊 Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')

        ### 🔹 STEP THE SCHEDULER ###
        if scheduler:
            scheduler.step()  

    print("🏁 Training Complete!")


 

def eval_model(model: nn.Module, dataloader: DataLoader, device='auto'):
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            if inputs.dim() == 3:  
                inputs = inputs.unsqueeze(0)  
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')

def get_predictions(model, dataloader, device):
    model.eval()
    y_true = []
    y_scores = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            if inputs.dim() == 3:  
                inputs = inputs.unsqueeze(0)  
            outputs = model(inputs)  
            probs = torch.softmax(outputs, dim=1)
            
            y_true.extend(labels.cpu().numpy())  
            y_scores.extend(probs.cpu().numpy())  
    
    return np.array(y_true), np.array(y_scores)    

def plot_multiclass_roc(model, dataloader, label_encoder, model_name=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    y_true, y_scores = get_predictions(model, dataloader, device)
    
    class_names = label_encoder.classes_  
    num_classes = len(class_names)

    y_true_one_hot = label_binarize(y_true, classes=np.arange(num_classes))
    
    fpr, tpr, roc_auc = {}, {}, {}
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    colors = cycle(["blue", "red", "green", "purple", "orange", "brown", "pink"])
    
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})") 

    plt.plot([0, 1], [0, 1], "k--", lw=2)  
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    title = "Multiclass ROC AUC"
    if model_name:
        title += f' {model_name}'
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def calculate_auc(model, dataloader, label_encoder):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    y_true, y_scores = get_predictions(model, dataloader, device)
    
    class_names = label_encoder.classes_
    num_classes = len(class_names)

    y_true_one_hot = label_binarize(y_true, classes=np.arange(num_classes))

    ovo_auc = roc_auc_score(y_true_one_hot, y_scores, multi_class='ovo')

    ovr_auc = roc_auc_score(y_true_one_hot, y_scores, multi_class='ovr')

    return ovo_auc, ovr_auc


def calculate_f1(model, dataloader, label_encoder):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    y_true, y_scores = get_predictions(model, dataloader, device)
    
    y_pred = np.argmax(y_scores, axis=1)
    
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    return f1


def calculate_accuracy(model, dataloader, device='auto'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(0)
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    return accuracy