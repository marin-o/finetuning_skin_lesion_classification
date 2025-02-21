# README: Skin Lesion Classification with HAM10000 Dataset

This repository contains code for training and evaluating a deep learning model for skin lesion classification using the HAM10000 dataset. The models are implemented from PyTorch, and the evaluation includes metrics like accuracy, F1 score, ROC AUC, and more. The following functions are available in the repository:

Functions
1. load_data

This function loads and preprocesses the data for both training and testing.
Parameters:

    train_metadata_path (str): Path to the CSV file containing metadata for the training dataset.
    test_metadata_path (str): Path to the CSV file containing metadata for the testing dataset.
    train_images_path (str): Path to the directory where the training images are stored.
    test_images_path (str): Path to the directory where the test images are stored.

Returns:

    train_df (pd.DataFrame): DataFrame containing the training data.
    val_df (pd.DataFrame): DataFrame containing the validation data.
    test_df (pd.DataFrame): DataFrame containing the testing data.

This function reads metadata and image paths for training and testing datasets, encodes the labels using LabelEncoder, and splits the training data into training and validation sets. It also visualizes the class distribution for both training and testing datasets.
2. show_image

This function visualizes a given image after reversing any normalization applied during preprocessing.
Parameters:

    image (Tensor): The image tensor to display.
    mean (float): The mean used for normalization.
    std (float): The standard deviation used for normalization.

3. train_model

This function trains the deep learning model for a given number of epochs.
Parameters:

    model (nn.Module): The deep learning model to train.
    train_dataloader (DataLoader): The DataLoader for the training dataset.
    val_dataloader (DataLoader): The DataLoader for the validation dataset.
    criterion (nn.Module): The loss function.
    optimizer (Optimizer): The optimizer.
    scheduler (lr_scheduler): The learning rate scheduler.
    device (torch.device): The device on which to run the model (CPU or GPU).
    num_epochs (int): The number of epochs for training.
    log_interval (int): The frequency of logging during training.

The function performs the training phase and validation phase for each epoch, printing the loss and accuracy.
4. eval_model

This function evaluates the model on the given test data.
Parameters:

    model (nn.Module): The trained model.
    dataloader (DataLoader): The DataLoader for the test dataset.
    device (torch.device): The device on which to evaluate the model (CPU or GPU).

It calculates and prints the accuracy of the model on the test set.
5. get_predictions

This function generates predictions for a given model on the provided dataset.
Parameters:

    model (nn.Module): The model to evaluate.
    dataloader (DataLoader): The DataLoader for the dataset.
    device (torch.device): The device on which to evaluate the model (CPU or GPU).

Returns:

    y_true (np.ndarray): The true labels for the dataset.
    y_scores (np.ndarray): The predicted probabilities for each class.

6. plot_multiclass_roc

This function plots the multiclass ROC curve for the given model.
Parameters:

    model (nn.Module): The trained model.
    dataloader (DataLoader): The DataLoader for the test dataset.
    label_encoder (LabelEncoder): The label encoder used for encoding the labels.
    model_name (str, optional): The name of the model to display in the title of the plot.

The function computes the ROC curve and AUC for each class and visualizes them on a single plot.
7. calculate_auc

This function calculates the One-Versus-One (OVO) and One-Versus-Rest (OVR) AUC for the model.
Parameters:

    model (nn.Module): The trained model.
    dataloader (DataLoader): The DataLoader for the test dataset.
    label_encoder (LabelEncoder): The label encoder used for encoding the labels.

Returns:

    ovo_auc (float): The AUC using the OVO strategy.
    ovr_auc (float): The AUC using the OVR strategy.

8. calculate_f1

This function calculates the F1 score (weighted average) for the model.
Parameters:

    model (nn.Module): The trained model.
    dataloader (DataLoader): The DataLoader for the test dataset.
    label_encoder (LabelEncoder): The label encoder used for encoding the labels.

Returns:

    f1 (float): The weighted average F1 score for the model.

9. calculate_accuracy

This function calculates the accuracy of the model on a given dataset.
Parameters:

    model (nn.Module): The trained model.
    dataloader (DataLoader): The DataLoader for the test dataset.
    device (torch.device): The device on which to evaluate the model (CPU or GPU).

Returns:

    accuracy (float): The accuracy of the model on the test dataset.

Example Usage
1. Loading the Data
```python
train_df, val_df, test_df = load_data(
    train_metadata_path='data/ham10000/HAM10000_metadata.csv',
    test_metadata_path='data/ham10000/images/ISIC2018_Task3_Test_GroundTruth.csv',
    train_images_path='data/ham10000/images/',
    test_images_path='data/ham10000/images/test_images/'
)
```
2. Training the Model
```python

train_model(
    model=my_model, 
    train_dataloader=train_loader, 
    val_dataloader=val_loader, 
    criterion=my_criterion, 
    optimizer=my_optimizer, 
    scheduler=my_scheduler, 
    device='cuda',
    num_epochs=10
)
```
3. Plotting ROC Curves
```python
plot_multiclass_roc(model=my_model, dataloader=test_loader, label_encoder=my_label_encoder, model_name="My Model")
```
4. Calculating AUC and F1 Score
```python
ovo_auc, ovr_auc = calculate_auc(model=my_model, dataloader=test_loader, label_encoder=my_label_encoder)
f1 = calculate_f1(model=my_model, dataloader=test_loader, label_encoder=my_label_encoder)
```

To install the required packages, you can use:

`pip install -r requirements.txt`