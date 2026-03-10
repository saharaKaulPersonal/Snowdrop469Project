import torch
import numpy as np

class CPWrapper:
    def __init__(self, model, alpha=0.1, scoring_fn=None, device='cpu'):
        self.model = model
        self.alpha = alpha
        self.device = device
        self.scoring_fn = scoring_fn
        self.threshold = None

    def fit(self, calib_data, batch_size=64):
        """Compute calibration scores and threshold.

        calib_data can be a DataLoader or a Dataset.
        """
        if isinstance(calib_data, torch.utils.data.DataLoader):
            calib_loader = calib_data
        else:
            calib_loader = torch.utils.data.DataLoader(
                calib_data, batch_size=batch_size, shuffle=False
            )

        self.model.eval()
        all_scores = []

        with torch.no_grad():
            for x, y in calib_loader:
                x, y = x.to(self.device), y.to(self.device)
                # Model already outputs a softmax vector
                probs = self.model(x).cpu().numpy()
                # Squeeze to (batch,) — datasets like MedMNIST return labels as (batch, 1) rather than (batch,)
                y_np = y.cpu().numpy().reshape(-1)
                scores = self.scoring_fn(probs, y_np)
                all_scores.extend(scores)

        all_scores = np.array(all_scores)
        n = len(all_scores)
        # (Section 1.1, Figure 1.2): ceil((n+1)(1-alpha))/n
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.threshold = np.quantile(all_scores, q, method='higher')
        return self

    def predict(self, x):
        """Return a conformal prediction set for each sample in x."""
        self.model.eval()
        x = x.to(self.device)

        with torch.no_grad():
            # Model already outputs a softmax vector
            probs = self.model(x).cpu().numpy()

        pred_sets = []
        for p in probs:
            scores = np.array([
                self.scoring_fn(p.reshape(1, -1), np.array([k]))[0]
                for k in range(len(p))
            ])

            included = np.where(scores <= self.threshold)[0]
            pred_sets.append(included.tolist())

        return pred_sets