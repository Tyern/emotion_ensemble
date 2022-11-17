import os
import matplotlib.pyplot as plt
import torch
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

def plot_training_process(acc, val_acc, loss, val_loss, save_path="./"):
    epochs = range(len(acc))

    plt.clf()
    plt.plot(epochs, acc, 'b', label='Training score', color="blue")
    plt.plot(epochs, val_acc, 'b', label='Validation score', color="orange")
    plt.title('Training and validation f1 score')
    plt.legend()
    plt.savefig(os.path.join(save_path, "acc.jpg"))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss', color="blue")
    plt.plot(epochs, val_loss, 'b', label='Validation loss', color="orange")
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, "loss.jpg"))

def save_confusion_matrix(class_names, y_pred_tensor, y_true_tensor):
    confmat = ConfusionMatrix(num_classes=len(class_names))
    confmat_tensor = confmat(
        preds=y_pred_tensor,
        target=y_true_tensor)
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy 
        class_names=class_names, # turn the row and column labels into class names
        figsize=(10, 7)
    )
    fig.savefig("confusion_matrix.jpg") 
    