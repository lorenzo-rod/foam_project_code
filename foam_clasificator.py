import torch
import torch.nn as nn
from regressor_plot import train_test_plot, train_test_plot_acc, compute_cf
from generate_plot_files import generate_plot_file
from calc_times import disp_elapsed_time, disp_remaining_time
import time
from train_test_gen import train_test_loaders
from nn_gen import design_model
import os


# Device configuration
# select GPU as computation device if it is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device is {device}")


# Hyperparameters
learning_rate = (1e-3)/2
hidden_size = 128*2 # number of neurons in the linear layers after the convolutional layers
n_epochs = 15 # number of epochs when training
batch_size = 23 # number of images in each batch
num_fc = 2 # number of linear layers in the output
weight_decay = 0 # weight decay in L2 reguralization
image_resize = 224 # size of the square images the input images will be resized to
model_to_use ="googlenet" # pretrained model to use (xception, resnet18, resnet50, resnet101)
layers_to_freeze = 0 # maximum is 24 for xception, maximum is 9 for resnet18
dropout_prob = 0 # dropout probability in dropout layers (these are in the linear layers) 
continue_training = False # wheter to continue training a model saved
class_names = ["bottom", "middle", "top"] # names of the different classes


# computes number of outputs in the final layers
n_outputs = len(class_names)


# Create the file where all plots will be saved if it does not exist
base_path = os.getcwd()
if not os.path.exists(os.path.join(base_path, "plot_clasificator")):
    os.makedirs(os.path.join(base_path, "plot_clasificator"))
path_plots = os.path.join(base_path, "plot_clasificator")


# Creates the file where plots will be saved fot this training session
path_plots = generate_plot_file(path_plots=path_plots, learning_rate=learning_rate, hidden_size=hidden_size, n_epochs=n_epochs,
 batch_size=batch_size, image_resize=image_resize, model_to_use=model_to_use, num_fc=num_fc,
  layers_to_freeze=layers_to_freeze, weight_decay=weight_decay, dropout_prob=dropout_prob)


# 0) Load Data
# Load the train dataset path
train_dataset_path = os.path.join(base_path, "datasets", "train_crops_299_clasif")
# Load the test dataset path
test_dataset_path = os.path.join(base_path, "datasets", "test_crops_299_clasif")
# Generates train and test loaders
train_loader, test_loader = train_test_loaders(train_dataset_path, test_dataset_path, batch_size, True, False, image_resize)


# 1) Design model (input, output size, forward pass)
# Design the model using the hyperparameters set above
model = design_model(model_to_use, hidden_size, layers_to_freeze, num_fc, n_outputs, dropout_prob)
# Load weights from memory if continue_training is True
if continue_training:
    PATH = ""
    model.load_state_dict(torch.load(PATH))
# Prints model structure and the number of trainable parameters
print(model)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters = {pytorch_total_params}")

# Sends the model to the device (CPU or GPU) and displays what is the pretrained model used
model.to(device)
print("Model is " + model_to_use)


# 2) Construct loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


# 3) Training loop
# 3.1) Epoch 0 (no training)
print("Starting epoch 0")
# Create arrays to save train and test losses and accuracies at every epoch
train_losses, test_losses, train_accs, test_accs = [], [], [], []
# Calculates total number of steps at every epoch
n_total_steps = len(train_loader)
# Sets Dropout layers and Batch Normalization layers to evaluation mode
model.eval()
# Sets requires_grad to False (gradient descent is not computed)
with torch.no_grad():
    # Epoch loss is initialized
    loss_average_train = 0
    for j, (images, labels) in enumerate(train_loader):
        # Images and labels are loaded
        images = images.to(device)
        labels = labels.to(device)
        # Model output is computed
        outputs = model(images)
        # Loss is calculated
        loss = criterion(outputs, labels)
        loss_average_train += loss.item()

# Epoch loss is computed
loss_average_train = loss_average_train / len(train_loader)

# Sets requires_grad to False (gradient descent is not computed)
with torch.no_grad():
    # Epoch loss is initialized
    loss_average_test = 0
    for j, (images, labels) in enumerate(test_loader):
        # Images and labels are loaded
        images = images.to(device)
        labels = labels.to(device)
        # Model output is computed
        outputs = model(images)
        # Loss is calculated
        loss = criterion(outputs, labels)
        loss_average_test += loss.item()

# Epoch loss is computed
loss_average_test = loss_average_test / len(test_loader)

# Train and test loss are saved in loss lists
train_losses.append(loss_average_train)
test_losses.append(loss_average_test)

# Training begins
# Model is saved
PATH = os.path.join(path_plots, model_to_use + '_lowest_test_loss_epoch' + '.pth')
torch.save(model.state_dict(), PATH)
# Lowest test loss value is initialized
lowest_test_loss = loss_average_test
# Lowest test loss epoch value is initialized
lowest_test_loss_epoch = 0
print("Training begins")
# Initial time is saved
time_init = time.time()
# Training loop begins
for epoch in range(n_epochs):
    # Number of samples and number of correct outputs are initialized
    n_correct = 0
    n_samples = 0
    # Time at which epoch started is saved
    time_init_epoch = time.time()
    # Epoch loss is initialized
    loss_average_train = 0
    # Sets Dropout layers and Batch Normalization layers to train mode
    model.train()
    # Epoch loop begins
    for i, (images, labels) in enumerate(train_loader):
        # Images and labels are loaded
        images = images.to(device)
        labels = labels.to(device)
        # forward pass: compute prediction
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_average_train += loss.item()
        _, predicted = torch.max(outputs, 1)
        # computation of batch accuraccy
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        train_acc = 100.0 * n_correct / n_samples
        # - backward pass: gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Data print (prints loss, epoch number, step number and accuracy every step)
        if (i+1) % 1 == 0:
            print (f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{n_total_steps}], Loss = {loss.item():.4f}, Accuracy = {train_acc:.4f}')

    # Epoch loss is calculated
    loss_average_train = loss_average_train / len(train_loader)


    # 4) Testing
    # Sets Dropout layers and Batch Normalization layers to evaluation mode
    model.eval()
    # Sets requires_grad to False (gradient descent is not computed)
    with torch.no_grad():
        # Number of samples and number of correct outputs are initialized
        n_correct = 0
        n_samples = 0
        # Test loss is initialized
        loss_average_test = 0
        # Test loop begins
        for j, (images, labels) in enumerate(test_loader):
            # Images and labels are loaded
            images = images.to(device)
            labels = labels.to(device)
            # Model output is computed
            outputs = model(images)
            # Loss is computed
            loss = criterion(outputs, labels)
            loss_average_test += loss.item()
            _, predicted = torch.max(outputs, 1)
            # computation of batch accuraccy
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        # Test accuracy is computed
        test_acc = 100.0 * n_correct / n_samples
        # Test loss is computed   
        loss_average_test = loss_average_test / len(test_loader)
        print(f"Test Loss = {loss_average_test:.4f}")
        print(f"Test acc = {test_acc:.4f}")
        # If the test loss is the lowest, the model is saved as the lowest test loss model
        if lowest_test_loss > loss_average_test:
            print("New minimum test loss")
            lowest_test_loss_epoch = epoch + 1
            lowest_test_loss = loss_average_test
            PATH = os.path.join(path_plots, model_to_use + '_lowest_test_loss_epoch' + '.pth')
            torch.save(model.state_dict(), PATH)


    # Train and test losses and accuracies are appended to their corresponding lists
    train_losses.append(loss_average_train)
    test_losses.append(loss_average_test)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    # Time at which the epoch was finished is saved
    time_fin = time.time()
    # Elapsed time is displayed
    disp_elapsed_time(time_fin=time_fin, time_init=time_init)
    # Remaining time is estimated and displayed
    disp_remaining_time(time_fin=time_fin, time_init_epoch=time_init_epoch, n_epochs=n_epochs, epoch=epoch)

# Final train loss, lowest test loss and the lowest test loss epoch are displayed
print(f"Final train loss = {train_losses[-1]}")
print(f"Lowest test loss = {lowest_test_loss}")
print(f"Lowest test loss epoch = {lowest_test_loss_epoch}")

# Generates losses and accuracy plots
train_test_plot(train_losses=train_losses, test_losses=test_losses, path=path_plots, model_name=model_to_use, fold=0)
train_test_plot_acc(train_acc=train_accs, test_acc=test_accs, path=path_plots, model_name=model_to_use, fold=0)


# Lowest test loss model is loaded
PATH = os.path.join(path_plots, model_to_use + '_lowest_test_loss_epoch.pth')
model = design_model(model_to_use, hidden_size, layers_to_freeze, num_fc, n_outputs, dropout_prob)
model.load_state_dict(torch.load(PATH))
# Sets Dropout layers and Batch Normalization layers to evaluation mode
model.eval()
model.to(device)

# Generates train and test loaders
train_loader, test_loader = train_test_loaders(train_dataset_path, test_dataset_path, batch_size, False, False, image_resize)
# Computes and saves train and test confusion matrixes for the lowest test loss model
compute_cf(train_loader=train_loader, test_loader=test_loader, device=device, model=model,
model_to_use=model_to_use, path_plots=path_plots, fold=0, n_classes=n_outputs, class_names=class_names)

# Adds the lowest test loss to the summary
with open(os.path.join(path_plots, "summary.txt"),"a") as fd:
    fd.write(f"Lowest test loss = {lowest_test_loss}")
