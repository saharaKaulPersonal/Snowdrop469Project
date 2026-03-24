import torch
import torch.nn.functional as F
import numpy as np

def oneminussoftmax(model_output, true_label):
    """
    Compute the 1 - softmax of the model output for the true label.

    Args:
        model_output (torch.Tensor): The output of the model (logits).
        true_label (int): The index of the true label.
    Returns:
        torch.Tensor: The computed 1 - softmax value for the true label.
    """
  
    # This is assuming the model output is not already passed through softmax
    #model_output = list(F.softmax(model_output, dim=0))
    # Get the softmax value for the true label
    batch_indices = torch.arange(model_output.shape[0])
    true_label_softmax = model_output[batch_indices, true_label]

    # Compute 1 - softmax for the true label
    result = 1 - true_label_softmax

    return result


def APS(model_output, true_label, regularized=False, lambda_r=0.05):
    """
    Compute the Average Precision Score (APS)

    Args:
        model_output (torch.Tensor): (batch, K)
        true_label (torch.Tensor): (batch,)
    Returns:
        torch.Tensor: (batch,)
    """

    batch_size = model_output.shape[0]
    scores = []

    for b in range(batch_size):

        probs = model_output[b]
        y = true_label[b].item()

        # create a one hot vector to keep track of where the true label went
        one_hot = np.zeros_like(probs)
        one_hot[y] = 1

        # sort both lists in the same way
        pairs = list(zip(probs, one_hot))
        pairs = sorted(pairs, key=lambda x: x[0], reverse=True)

        # calculate score
        aps_score = pairs[0][0]
        i = 0
        while pairs[i][1] != 1:
            i += 1
            aps_score += pairs[i][0]

        if regularized:
            aps_score += i*lambda_r

        scores.append(float(aps_score))

    return torch.tensor(scores)


def margin(model_output, true_label):
    """
    Compares the best true class with the true label 

    Args:
        model_output (torch.Tensor): The output of the model
        true_label (int): The index of the true label
    Returns:
        Tensor: The computed margin score for the true label .
    """
    if isinstance(model_output, np.ndarray):
        model_output = torch.tensor(model_output)

    if isinstance(true_label, np.ndarray):
        true_label = torch.tensor(true_label)
        
    batch_size = model_output.shape[0]
    true_values = model_output[torch.arange(batch_size), true_label]
    mo = model_output.clone()
    mo[torch.arange(batch_size), true_label] = float('-inf')
    max_val = mo.max(dim=1).values

    return max_val - true_values

