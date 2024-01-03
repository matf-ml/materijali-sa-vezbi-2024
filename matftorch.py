import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split

def get_device():
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def bind_gpu(data):
  device = get_device()
  if isinstance(data, (list, tuple)):
    return [bind_gpu(data_elem) for data_elem in data]
  else:
    return data.to(device, non_blocking=True)

def scale_data(x_train, x_test):
  scaler = StandardScaler()
  scaler.fit(x_train)
  x_train = scaler.transform(x_train)
  x_test = scaler.transform(x_test)
  return scaler, x_train, x_test

def count_parameters(model):
  total_params = sum(p.numel() for p in model.parameters())

  # Iterate through the layers and print their details
  for name, layer in model.named_children():
        num_params = sum(p.numel() for p in layer.parameters())
        print(f"Layer: {name}, Parameters: {num_params}")

  return total_params

def get_data_loader(X_data, y_data, batch_size, shuffle):
  """
    Get torch data loader from the DataFrame and Series objects.
  """
  X_tensor = torch.FloatTensor(X_data)
  if isinstance(y_data, pd.Series):
    y_data = y_data.values
  y_tensor = torch.FloatTensor(y_data)
  dataset = TensorDataset(X_tensor, y_tensor)
  return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_classification(model, criterion, optimizer, number_of_epochs, train_loader, multiclass=False):
  losses = []
  accuracies = []
  device = get_device()

  for epoch in range(number_of_epochs):
      running_loss = 0.0
      correct = 0
      total = 0

      for inputs, labels in train_loader:
          inputs, labels = inputs.to(device), labels.to(device)
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs.squeeze(), labels)
          loss.backward()
          optimizer.step()

          running_loss += loss.item()

          if multiclass:
            predicted = torch.argmax(outputs, dim=1)
          else:
            predicted = (outputs > 0.5).float()
          correct += (predicted.squeeze() == labels).sum().item()
          total += labels.size(0)

      epoch_loss = running_loss / len(train_loader)
      epoch_accuracy = correct / total
      losses.append(epoch_loss)
      accuracies.append(epoch_accuracy)
      print(f"Epoch [{epoch + 1}/{number_of_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

  return losses, accuracies

def plot_classification(loss, accuracy, loss_title='Training Loss', accuracy_title='Training Accuracy'):
  number_of_epochs = len(loss)

  plt.title(loss_title)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.plot(range(number_of_epochs), loss)
  plt.show()

  plt.title(accuracy_title)
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.plot(range(number_of_epochs), accuracy)
  plt.show()

def evaluate_classification(model, criterion, loader, multiclass=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    predicted_labels, true_labels = [], []
    device = get_device()

    with torch.no_grad():  # No gradient computation during evaluation
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs.squeeze(), labels).item()

            if multiclass:
              predicted = torch.argmax(outputs, dim=1)
            else:
              predicted = (outputs > 0.5).float()
            predicted_labels.extend(predicted.squeeze().tolist())
            true_labels.extend(labels.tolist())

            total_samples += labels.size(0)
            total_correct += (predicted.squeeze() == labels).sum().item()

    # compute metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
    print(f'Model evaluation on: {loader}')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    # plot
    cm = confusion_matrix(true_labels, predicted_labels)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

# Training and plotting that uses validation data
def train_classification(model, criterion, optimizer, number_of_epochs, train_loader, validation_loader, multiclass=False):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    device = get_device()

    for epoch in range(number_of_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if multiclass:
              predicted = torch.argmax(outputs, dim=1)
            else:
              predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_accuracy = correct / total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)
        print(f"Epoch [{epoch + 1}/{number_of_epochs}], Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}")

        if validation_loader:
            model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for val_inputs, val_labels in validation_loader:
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    val_outputs = model(val_inputs)
                    val_loss = criterion(val_outputs.squeeze(), val_labels)
                    val_running_loss += val_loss.item()

                    if multiclass:
                      val_predicted = torch.argmax(val_outputs, dim=1)
                    else:
                      val_predicted = (val_outputs > 0.5).float()
                    val_total += val_labels.size(0)
                    val_correct += (val_predicted.squeeze() == val_labels).sum().item()

            epoch_val_loss = val_running_loss / len(validation_loader)
            epoch_val_accuracy = val_correct / val_total
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_accuracy)
            print(f"Epoch [{epoch + 1}/{number_of_epochs}], Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_accuracy:.4f}")

    return train_losses, train_accuracies, val_losses, val_accuracies

def plot_classification(train_loss, train_accuracy, val_loss, val_accuracy):
    number_of_epochs = len(train_loss)
    epochs = range(1, number_of_epochs + 1)

    plt.figure(figsize=(12, 5))

    # Plotting Training Loss
    plt.subplot(1, 2, 1)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, train_loss, label='Training Loss')
    if val_loss:
        plt.plot(epochs, val_loss, label='Validation Loss')

    plt.legend()

    # Plotting Training Accuracy
    plt.subplot(1, 2, 2)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accuracy, label='Training Accuracy')
    if val_accuracy:
        plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Regression models
def train_regression(model, criterion, optimizer, number_of_epochs, train_loader, validation_loader=None, steps_per_train_epoch=None, val_steps=None):
    train_losses, val_losses = [], []
    device = get_device()

    for epoch in range(number_of_epochs):
        model.train()
        running_loss = 0.0
        batches_processed = 0

        for inputs, targets in train_loader:
            if steps_per_train_epoch is not None and batches_processed >= steps_per_train_epoch:
                break  # Exit the loop if the desired number of steps is reached

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batches_processed += 1

        epoch_train_loss = running_loss / batches_processed if steps_per_train_epoch is not None else running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        print(f"Epoch [{epoch + 1}/{number_of_epochs}], Train Loss: {epoch_train_loss:.4f}")

        if validation_loader:
            model.eval()
            val_running_loss = 0.0
            val_batches_processed = 0

            with torch.no_grad():
                for val_inputs, val_targets in validation_loader:
                    if val_steps is not None and val_batches_processed >= val_steps:
                        break  # Exit the loop if the desired number of validation steps is reached

                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                    val_outputs = model(val_inputs)
                    val_loss = criterion(val_outputs.squeeze(), val_targets)
                    val_running_loss += val_loss.item()
                    val_batches_processed += 1

            epoch_val_loss = val_running_loss / val_batches_processed if val_steps is not None else val_running_loss / len(validation_loader)
            val_losses.append(epoch_val_loss)
            print(f"Epoch [{epoch + 1}/{number_of_epochs}], Validation Loss: {epoch_val_loss:.4f}")

    return train_losses, val_losses

def evaluate_regression(model, criterion, loader, test_steps=None):
    model.eval()
    total_loss = 0.0
    device = get_device()
    predicted_labels, true_labels = [], []
    batches_processed = 0

    with torch.no_grad():  # No gradient computation during evaluation
        for inputs, labels in loader:
            if test_steps is not None and batches_processed >= test_steps:
                break  # Exit the evaluation loop if the desired number of test steps is reached

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()

            predicted_labels.extend(outputs.squeeze().cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            batches_processed += 1

    # Compute regression metrics
    avg_loss = total_loss / batches_processed if test_steps is not None else total_loss / len(loader)
    mse = mean_squared_error(true_labels, predicted_labels)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_labels, predicted_labels)
    r2 = r2_score(true_labels, predicted_labels)

    print(f'Model evaluation on: {loader}')
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return avg_loss, mse, rmse, mae, r2

def plot_regression(train_losses, val_losses):
    epochs = len(train_losses)
    epochs_range = range(epochs)

    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss', linestyle='--', color='orange')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()