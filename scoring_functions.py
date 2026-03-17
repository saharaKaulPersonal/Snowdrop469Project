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

def APS(model_output, true_label):
    """
    model_output : (batch, K) numpy array of probabilities
    true_label   : (batch,) numpy array
    """

    batch_size = model_output.shape[0]
    scores = []

    for b in range(batch_size):

        probs = model_output[b]
        y = true_label[b]

        # sort probabilities descending
        order = np.argsort(-probs)
        sorted_probs = probs[order]

        # find rank of true label
        rank = np.where(order == y)[0][0]

        # cumulative probability up to that rank
        aps_score = np.sum(sorted_probs[:rank+1])

        scores.append(aps_score)

    return np.array(scores)

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
        model_output = torch.from_numpy(model_output)
    if isinstance(true_label, np.ndarray):
        true_label = torch.from_numpy(true_label)
    
    batch_size = model_output.shape[0]
    true_values = model_output[torch.arange(batch_size), true_label]
    mo = model_output.clone()
    mo[torch.arange(batch_size), true_label] = float('-inf')
    max_val = mo.max(dim=1).values

    return max_val - true_values