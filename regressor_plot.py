import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os


def train_test_plot(train_losses, test_losses, path, model_name, fold):
    """Generates test loss vs epochs and train loss vs epochs figure and saves it
    in path

    Args:
        train_losses (list): list of floats containing the train losses for every epoch
        test_losses (list): list of floats containing the test losses for every epoch
        path (str): path where the figure will be saved
        model_name (str): pretrained model used when training
        fold (int): Fold number (used only in cross k validation)
    """

    plot2 = plt.figure(2)
    plt.plot(train_losses,'-o')
    plt.plot(test_losses,'-o')
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train','Validation'])
    plt.title('Train vs Test Losses')
    plt.grid()
    plt.savefig(os.path.join(path, model_name + '_train_test_loss_' + str(fold) + '.png'))
    plot2.clf()

def train_test_plot_acc(train_acc, test_acc, path, model_name, fold):

    plot2 = plt.figure(2)
    plt.plot(train_acc,'-o')
    plt.plot(test_acc,'-o')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train','Test'])
    plt.title('Train vs Test accuracies')
    plt.grid()
    plt.savefig(os.path.join(path, model_name + '_train_test_accuracy_' + str(fold) + '.png'))
    plot2.clf()

def compute_cf(train_loader, test_loader, device, model,
    model_to_use, path_plots, n_classes, class_names, fold=0):

    with torch.no_grad():
        # confusion matrix initialization
        row_matrix = [0 for i in range(n_classes)]
        confusion_mat = []
        [confusion_mat.append(row_matrix) for i in range(n_classes)]
        for images, labels in test_loader:
            # Images and labels are loaded
            images = images.to(device)
            labels = labels.to(device)
            # Model output is computed
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            # Confusion Matrix is updated
            confusion_mat += confusion_matrix(labels.cpu(), predicted.cpu(), labels=[i for i in range(n_classes)])
    
    # Test Confusion Matrix is saved and displayed
    plot1 = plt.figure(1)
    disp = ConfusionMatrixDisplay(confusion_mat, display_labels=class_names)
    disp.plot()
    plt.savefig(os.path.join(path_plots, model_to_use + "confusion_matrix_test" + str(fold) + ".png"))
    plot1.clf()

    with torch.no_grad():
        # confusion matrix initialization
        row_matrix = [0 for i in range(n_classes)]
        confusion_mat = []
        [confusion_mat.append(row_matrix) for i in range(n_classes)]
        for images, labels in train_loader:
            # Images and labels are loaded
            images = images.to(device)
            labels = labels.to(device)
            # Model output is computed
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            # Confusion Matrix is updated
            confusion_mat += confusion_matrix(labels.cpu(), predicted.cpu(), labels=[i for i in range(n_classes)])
    
    # Train Confusion Matrix is saved and displayed
    plot1 = plt.figure(1)
    disp = ConfusionMatrixDisplay(confusion_mat, display_labels=class_names)
    disp.plot()
    plt.savefig(os.path.join(path_plots, model_to_use + "confusion_matrix_train" + str(fold) + ".png"))
    plot1.clf()


def labels_vs_outputs(labels_list, outputs_list, path, model_name, fold):
    """Saves a plot with arrays labels_list and outputs_list in path

    Args:
        labels_list (list): list containing the labels of the images
        outputs_list (list): list containing the outputs of the model
        path (str): path where the plot will be saved
        model_name (str): Pretrained model used
    """
    plot3 = plt.figure(3)
    plt.plot(labels_list,'-o')
    plt.plot(outputs_list,'o')
    plt.ylabel('')
    plt.legend(['Label','Output'])
    plt.title('True labels vs model output')
    plt.grid()
    plt.savefig(os.path.join(path, model_name + '_label_output_' + str(fold) + '.png'))
    plot3.clf()


def gen_regression_plots(train_loader, test_loader, device, model, model_to_use, criterion, path_plots, fold):
    """generates plots that have the train and test labels vs the model output

    Args:
        train_loader (DataLoader): train set loader
        test_loader (DataLoader): test set loader
        device (str): device in which to carry operations (cpu or cuda)
        model (nn.Module): neural network used to compute the plots
        model_to_use (str): trained model used
        criterion (criterion): loss criterion
        path_plots (str): path where the plots will be saved
    
    Returns:
        loss_test_average (float): test loss calculated using the graph
    """
    with torch.no_grad():
        outputs_list = []
        labels_list = []
        loss_average_train_qq = 0
        for j, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.to(dtype=torch.float32)
            outputs = model(images)
            outputs = outputs.reshape((-1,))
            loss = criterion(outputs, labels)
            loss_average_train_qq += loss.item()
            [outputs_list.append(i.item()) for i in outputs]
            [labels_list.append(i.item()) for i in labels]

    loss_average_train_qq = (loss_average_train_qq / len(train_loader))


    print(f"train_lost_qq = {loss_average_train_qq}")
    labels_vs_outputs(labels_list=labels_list, outputs_list=outputs_list, path=path_plots, model_name=(model_to_use + "_train"), fold=fold)


    with torch.no_grad():
        outputs_list = []
        labels_list = []
        loss_test_average = 0
        for j, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.to(dtype=torch.float32)
            outputs = model(images)
            outputs = outputs.reshape((-1,))
            loss = criterion(outputs, labels)
            loss_test_average += loss.item()
            [outputs_list.append(i.item()) for i in outputs]
            [labels_list.append(i.item()) for i in labels]

    loss_test_average = (loss_test_average / len(test_loader))


    labels_vs_outputs(labels_list=labels_list, outputs_list=outputs_list, path=path_plots, model_name=model_to_use, fold=fold)
    
    return loss_test_average
