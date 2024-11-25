import os
import torch
import logging
from metrics.cls import calculate_classification_metrics


def evaluate_and_save_model(epoch, model, test_loader, device, args):
    """
    Evaluates the model on the test set, calculates metrics, and saves the model state_dict if certain conditions are met.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        test_loader (torch.utils.data.DataLoader): The data loader for the test set.
        device (torch.device): The device on which to run the model.
        args (argparse.Namespace): Contains arguments such as model name, fold number, snapshot directory, etc.
        calculate_classification_metrics (callable): A function that calculates classification metrics.

    Returns:
        metrics_avg (list): A list containing the average metrics (accuracy, F1-score, precision, recall, kappa).
    """
    model.eval()  # Switch to evaluation mode

    # # Prepare directory for saving the model
    # model_l_savedir = os.path.join(args.snapshot_dir, f'model_{args.model}_fold{args.fold}')
    # if not os.path.exists(model_l_savedir):
    #     os.makedirs(model_l_savedir)
    #
    #     # Save the model
    # torch.save(model.state_dict(), os.path.join(model_l_savedir, f'fold{args.fold}_epoch{epoch}_{args.model}.pth'))

    # Initialize metrics sum
    metrics_sum = [0.0 for _ in range(5)]

    # Evaluate the model
    for step, batch in enumerate(test_loader):
        images, labels = batch
        images = images.to(device)
        labels = labels.long().to(device)

        with torch.no_grad():
            outs = model(images)

            # Assuming calculate_classification_metrics takes raw output and labels
            # Adjust output processing as needed (e.g., argmax for classification)
        output = outs.cpu().numpy()
        label = labels.cpu().numpy()
        metrics_dict = calculate_classification_metrics(output, label)

        # Sum up metrics
        metrics_sum = [sum_val + metrics_dict[metric] for sum_val, metric in zip(metrics_sum,
                                                                                     ['accuracy', 'f1-score',
                                                                                      'precision',
                                                                                      'recall', 'kappa'])]

        # Calculate average metrics
    metrics_avg = [sum_val / len(test_loader) for sum_val in metrics_sum]

    # Optionally, switch back to train mode if needed (not strictly necessary here)
    model.train()

    return metrics_avg

# Note: This function assumes 'epoch' is defined in the surrounding scope.
# In practice, you might want to pass 'epoch' as an argument to this function.