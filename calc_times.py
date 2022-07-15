

def disp_elapsed_time(time_fin, time_init):
    """Displays elapsed time between time_init and time_fin in minutes, seconds format

    Args:
        time_fin (float): End time
        time_init (float): Initial time
    """
    minutes = int((time_fin - time_init) / 60)
    seconds = time_fin - time_init - 60 * minutes
    if seconds < 0:
        seconds = 0
    seconds = int(seconds)
    print(f"Time elapsed: {minutes} minutes, {seconds} seconds")


def disp_remaining_time(time_fin, time_init_epoch, n_epochs, epoch):
    """Estimates and displays time remaining when training a model

    Args:
        time_fin (float): time at which last epoch was finished
        time_init_epoch (float): time at which last epoch begun
        n_epochs (int): Total number of epochs
        epoch (int): Current epoch number
    """
    time_remaining = (time_fin - time_init_epoch)*(n_epochs - epoch - 1)
    minutes = int(time_remaining / 60)
    seconds = time_remaining - 60 * minutes
    if seconds < 0:
        seconds = 0
    seconds = int(seconds)
    print(f"Estimated time remaining: {minutes} minutes, {seconds} seconds")