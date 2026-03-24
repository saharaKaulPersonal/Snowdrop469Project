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
    
    batch_size = model_output.shape[0]
    true_values = model_output[torch.arange(batch_size), true_label]
    mo = model_output.clone()
    mo[torch.arange(batch_size), true_label] = float('-inf')
    max_val = mo.max(dim=1).values

    return max_val - true_values



# -------- COLON PATHOLOGY --------
# Classes and max penalty incurred if missed (worst-case row value):
#   0: Colorectal Adenocarcinoma Epithelium  (max penalty: 10.0)
#   1: Cancer Associated Stroma              (max penalty:  9.0)
#   2: Lymphocytes                           (max penalty:  6.5)
#   3: Normal Colon Mucosa                   (max penalty:  7.5)
#   4: Smooth Muscle                         (max penalty:  7.5)
#   5: Adipose                               (max penalty:  7.5)
#   6: Mucus                                 (max penalty:  8.5)
#   7: Debris                                (max penalty: 10.0)
#   8: Background                            (max penalty: 10.0)
#
# Penalty logic: higher penalty when true class is high severity
#                but model predicts a low-severity class.

COLON_CM = torch.tensor([
#  Pred→    0     1     2     3     4     5     6     7     8
    [0.0,  2.5,  5.0,  7.5,  7.5,  7.5,  8.5,  10.0, 10.0],  # True 0: Colorectal Adenocarcinoma Epithelium 
    [2.5,  0.0,  3.5,  6.0,  6.0,  6.0,  7.5,  9.0,  9.0 ],  # True 1: Cancer Associated Stroma
    [5.0,  3.5,  0.0,  3.5,  3.5,  3.5,  5.0,  6.5,  6.5 ],  # True 2: Lymphocytes
    [7.5,  6.0,  3.5,  0.0,  2.5,  2.5,  3.5,  5.0,  5.0 ],  # True 3: Normal Colon Mucosa
    [7.5,  6.0,  3.5,  2.5,  0.0,  1.5,  2.5,  4.0,  4.0 ],  # True 4: Smooth Muscle
    [7.5,  6.0,  3.5,  2.5,  1.5,  0.0,  2.5,  4.0,  4.0 ],  # True 5: Adipose
    [8.5,  7.5,  5.0,  3.5,  2.5,  2.5,  0.0,  2.5,  2.5 ],  # True 6: Mucus
    [10.0, 9.0,  6.5,  5.0,  4.0,  4.0,  2.5,  0.0,  1.5 ],  # True 7: Debris
    [10.0, 9.0,  6.5,  5.0,  4.0,  4.0,  2.5,  1.5,  0.0 ],  # True 8: Background
], dtype=torch.float32)
 
 
# -------- RETINAL OCT --------
# Classes and max penalty incurred if missed (worst-case row value):
#   0: Normal  (max penalty:  7.0)
#   1: Drusen  (max penalty:  8.0)
#   2: DME     (max penalty:  9.0)
#   3: CNV     (max penalty: 10.0)
#
# CNV missed as Normal = most dangerous mistake in this dataset.

OCT_CM = torch.tensor([
#  Pred→    0      1     2     3
    [0.0,   2.0,  5.0,  7.0],  # True 0: Normal
    [2.0,   0.0,  4.0,  8.0],  # True 1: Drusen
    [5.0,   4.0,  0.0,  9.0],  # True 2: DME
    [10.0,  8.0,  6.0,  0.0],  # True 3: CNV 
], dtype=torch.float32)
 
 
# -------- KIDNEY CORTEX HISTOLOGY --------
# Classes and max penalty incurred if missed (worst-case row value):
#   0: Podocytes                    (max penalty: 10.0)
#   1: Glomerular Endothelial Cells (max penalty: 10.0)
#   2: Leukocytes                   (max penalty:  7.0)
#   3: Interstitial Endothelial     (max penalty:  7.0)
#   4: Proximal Tubule              (max penalty:  7.0)
#   5: Thick Ascending Limb         (max penalty:  8.5)
#   6: Distal Tubule                (max penalty:  8.5)
#   7: Collecting Duct              (max penalty: 10.0)
#
# Glomerular classes missed as tubular = most dangerous mistake.

KIDNEY_CM = torch.tensor([
#  Pred→    0      1     2     3     4     5     6     7
    [0.0,   2.5,  5.5,  5.5,  7.0,  8.5,  8.5,  10.0],  # True 0: Podocytes     
    [2.5,   0.0,  5.5,  5.5,  7.0,  8.5,  8.5,  10.0],  # True 1: Glomerular Endothelial Cells 
    [5.5,   5.5,  0.0,  2.5,  4.0,  5.5,  5.5,  7.0 ],  # True 2: Leukocytes
    [5.5,   5.5,  2.5,  0.0,  4.0,  5.5,  5.5,  7.0 ],  # True 3: Interstitial Endothelial
    [7.0,   7.0,  4.0,  4.0,  0.0,  2.5,  2.5,  4.0 ],  # True 4: Proximal Tubule
    [8.5,   8.5,  5.5,  5.5,  2.5,  0.0,  2.5,  4.0 ],  # True 5: Thick Ascending Limb
    [8.5,   8.5,  5.5,  5.5,  2.5,  2.5,  0.0,  2.5 ],  # True 6: Distal Tubule
    [10.0,  10.0, 7.0,  7.0,  4.0,  4.0,  2.5,  0.0 ],  # True 7: Collecting Duct
], dtype=torch.float32)
 

 
def general_loss_path(model_output, true_label, loss_function, alpha=0.5):
    """
    Colon pathology general loss with soft confusion-matrix penalty.
 
    Args:
        model_output  (Tensor): logits, shape (batch, 9)
        true_label    (Tensor): ground truth class indices, shape (batch,)
        loss_function (callable): base loss fn, signature f(max_wrong, true_val) -> Tensor (batch,)
        alpha         (float): weight for the confusion penalty term
 
    Returns:
        Tensor: per-sample loss, shape (batch,)
    """
    batch_size = model_output.shape[0]
    cm = COLON_CM.to(model_output.device)
 
    true_values = model_output[torch.arange(batch_size), true_label]
 
    mo = model_output.clone()
    mo[torch.arange(batch_size), true_label] = float('-inf')
    max_val = mo.max(dim=1).values
 
    base_loss = loss_function(max_val, true_values)
 
    probs = F.softmax(model_output, dim=1)
    penalties = (probs * cm[true_label]).sum(dim=1)
 
    return base_loss + alpha * penalties
 
 
def general_loss_oct(model_output, true_label, loss_function, alpha=0.5):
    """
    Retinal OCT general loss with soft confusion-matrix penalty.
 
    Args:
        model_output  (Tensor): logits, shape (batch, 4)
        true_label    (Tensor): ground truth class indices, shape (batch,)
        loss_function (callable): base loss fn, signature f(max_wrong, true_val) -> Tensor (batch,)
        alpha         (float): weight for the confusion penalty term
 
    Returns:
        Tensor: per-sample loss, shape (batch,)
    """
    batch_size = model_output.shape[0]
    cm = OCT_CM.to(model_output.device)
 
    true_values = model_output[torch.arange(batch_size), true_label]
 
    mo = model_output.clone()
    mo[torch.arange(batch_size), true_label] = float('-inf')
    max_val = mo.max(dim=1).values
 
    base_loss = loss_function(max_val, true_values)
 
    probs = F.softmax(model_output, dim=1)
    penalties = (probs * cm[true_label]).sum(dim=1)
 
    return base_loss + alpha * penalties
 
 
def general_loss_tissue(model_output, true_label, loss_function, alpha=0.5):
    """
    Kidney cortex tissue general loss with soft confusion-matrix penalty.
 
    Args:
        model_output  (Tensor): logits, shape (batch, 8)
        true_label    (Tensor): ground truth class indices, shape (batch,)
        loss_function (callable): base loss fn, signature f(max_wrong, true_val) -> Tensor (batch,)
        alpha         (float): weight for the confusion penalty term
 
    Returns:
        Tensor: per-sample loss, shape (batch,)
    """
    batch_size = model_output.shape[0]
    cm = KIDNEY_CM.to(model_output.device)
 
    true_values = model_output[torch.arange(batch_size), true_label]
 
    mo = model_output.clone()
    mo[torch.arange(batch_size), true_label] = float('-inf')
    max_val = mo.max(dim=1).values
 
    base_loss = loss_function(max_val, true_values)
 
    probs = F.softmax(model_output, dim=1)
    penalties = (probs * cm[true_label]).sum(dim=1)
 
    return base_loss + alpha * penalties