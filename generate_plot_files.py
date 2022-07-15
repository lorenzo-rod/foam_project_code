import os

def generate_plot_file(path_plots, learning_rate, hidden_size, n_epochs, batch_size, image_resize, model_to_use, num_fc,
 layers_to_freeze, weight_decay, dropout_prob):
    """generates the file in which the plots will be stored and creates a .txt with the specifications of the training

    Args:
        path_plots (str): path where the file will be created
        learning_rate (float): learning rate used when training
        hidden_size (int): hidden size of the linear layers in the neural network
        n_epochs (int): number of epochs used when training
        batch_size (int): batch size used when training
        image_resize (int): size used when resizing the image to a square image
        model_to_use (str): pretrained model used
        num_fc (int): number of linear layers after the cnns
        layers_to_freeze (int): number of layers to freeze
        weight_decay (float): weight decay used in L2 Reguralization
        dropout_prob (float): droput probability used in droput layers

    Returns:
        path_plots (str): created file path
    """
    count = 0
    path_plots_original = path_plots
    path_plots = os.path.join(path_plots, "plot" + str(count))

    if not os.path.exists(path_plots):
        os.makedirs(path_plots)
        with open(path_plots+"/summary.txt","w") as fd:
            fd.write(f"learning rate = {learning_rate}\n")
            fd.write(f"hidden size = {hidden_size}\n")
            fd.write(f"n_epochs = {n_epochs}\n")
            fd.write(f"batch size = {batch_size}\n")
            fd.write(f"image resize = {image_resize}\n")
            fd.write(f"model = {model_to_use}\n")
            fd.write(f"Num FC = {num_fc}\n")
            fd.write(f"Frozen layers = {layers_to_freeze}\n")
            fd.write(f"Weight decay = {weight_decay}\n")
            fd.write(f"Droput Probability = {dropout_prob}\n")
    else:
        while(os.path.exists(path_plots)):
            count += 1
            path_plots = os.path.join(path_plots_original, "plot" + str(count))
            if not os.path.exists(path_plots):
                os.makedirs(path_plots)
                with open(path_plots+"/summary.txt","w") as fd:
                    fd.write(f"learning rate = {learning_rate}\n")
                    fd.write(f"hidden size = {hidden_size}\n")
                    fd.write(f"n_epochs = {n_epochs}\n")
                    fd.write(f"batch size = {batch_size}\n")
                    fd.write(f"image resize = {image_resize}\n")
                    fd.write(f"model = {model_to_use}\n")
                    fd.write(f"Num FC = {num_fc}\n")
                    fd.write(f"Frozen layers = {layers_to_freeze}\n")
                    fd.write(f"Weight decay = {weight_decay}\n")
                    fd.write(f"Droput Probability = {dropout_prob}\n")
                break
    
    return path_plots
