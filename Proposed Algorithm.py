import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Gaussian Normalization Module
# ------------------------------
class GaussianNormalization(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = None
        self.std = None

    def forward(self, x):
        if self.mean is None or self.std is None:
            self.mean = x.mean(dim=0, keepdim=True)
            self.std = x.std(dim=0, keepdim=True) + 1e-6
        x_norm = (x - self.mean) / self.std
        return x_norm

# ------------------------------
# Plant Leaf Disease Model
# ------------------------------
class PlantLeafDiseaseModel(nn.Module):
    def __init__(self, input_dim, num_classes=2, label_smoothing=0.1):
        super().__init__()
        self.gaussian_norm = GaussianNormalization()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, num_classes)
        self.label_smoothing = label_smoothing

    def forward(self, x):
        x = self.gaussian_norm(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.output(x)
        return logits

    def loss_fn(self, logits, targets):
        num_classes = logits.size(-1)
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
            smooth_targets = smooth_targets * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(smooth_targets * log_probs).sum(dim=1).mean()
        return loss

# ------------------------------
# Self-Adaptive RSA Optimizer
# ------------------------------
class SelfAdaptiveRSA:
    def __init__(self, n_dim, n_pop, lb, ub, max_iter, alpha_max=1.0, alpha_min=0.1, epsilon=1e-8):
        self.n_dim = n_dim
        self.n_pop = n_pop
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.max_iter = max_iter
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.epsilon = epsilon
        self.X = self.lb + (self.ub - self.lb) * np.random.rand(n_pop, n_dim)
        self.best_pos = np.zeros(n_dim)
        self.best_fit = float('inf')

    def evaluate(self, solution):
        return np.sum(solution**2)

    def population_diversity(self):
        std_dev = np.std(self.X, axis=0)
        diversity = np.sum(std_dev) / (np.sum(self.ub - self.lb) + self.epsilon)
        return np.clip(diversity, 0, 1)

    def run(self):
        for t in range(1, self.max_iter + 1):
            fitnesses = np.array([self.evaluate(ind) for ind in self.X])
            best_idx = np.argmin(fitnesses)
            if fitnesses[best_idx] < self.best_fit:
                self.best_fit = fitnesses[best_idx]
                self.best_pos = self.X[best_idx].copy()

            D_t = self.population_diversity()
            alpha_t = (self.alpha_max - (t / self.max_iter) * (self.alpha_max - self.alpha_min)) * (1 - D_t)
            M_x = np.mean(self.X, axis=1)

            for i in range(self.n_pop):
                for j in range(self.n_dim):
                    r1, r2 = np.random.choice([idx for idx in range(self.n_pop) if idx != i], 2, replace=False)
                    rand_val = np.random.rand()
                    P_ij = alpha_t + (self.X[i, j] - M_x[i]) / (self.best_pos[j] * (self.ub[j] - self.lb[j]) + self.epsilon)
                    eta_ij = self.best_pos[j] * P_ij
                    R_ij = (self.best_pos[j] - self.X[r2, j]) / (self.best_pos[j] + self.epsilon)

                    if t <= self.max_iter / 4:
                        beta = 0.5
                        self.X[i, j] = self.best_pos[j] * (-eta_ij * beta - R_ij * rand_val)
                    elif self.max_iter / 4 < t <= self.max_iter / 2:
                        r3 = np.random.uniform(-1, 1)
                        ES_t = 2 * r3 * (1 - t / self.max_iter)
                        self.X[i, j] = self.best_pos[j] * self.X[r1, j] * ES_t * rand_val
                    elif self.max_iter / 2 < t <= 3 * self.max_iter / 4:
                        self.X[i, j] = self.best_pos[j] * P_ij * rand_val
                    else:
                        self.X[i, j] = self.best_pos[j] - eta_ij * alpha_t - R_ij * rand_val

                self.X[i] = np.clip(self.X[i], self.lb, self.ub)

        return self.best_pos, self.best_fit

# ------------------------------
# Main Function
# ------------------------------
if __name__ == "__main__":
    # Load Data
    csv_path = '/content/drive/MyDrive/Colab Notebooks/Plant Dieases/Test/concadenate fused_file1.csv'
    df = pd.read_csv(csv_path)

    df_features = df.drop(columns=['filename', 'class'], errors='ignore')
    numeric_features = df_features.select_dtypes(include=[np.number])
    print(f"Using {numeric_features.shape[1]} numeric features for training.")

    X = numeric_features.values
    y = df['class'].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_encoded))
    model = PlantLeafDiseaseModel(input_dim=input_dim, num_classes=num_classes)

    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        predictions = torch.argmax(logits, dim=1)

    y_true = y_test.numpy()
    y_pred = predictions.numpy()

    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    fmeasure = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - (tp + fp + fn)

    sensitivity = np.mean(tp / (tp + fn + 1e-6))
    specificity = np.mean(tn / (tn + fp + 1e-6))
    npv = np.mean(tn / (tn + fn + 1e-6))
    fpr = np.mean(fp / (fp + tn + 1e-6))
    fnr = np.mean(fn / (fn + tp + 1e-6))

    print(f"\nAccuracy: {accuracy:.4f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F-measure: {fmeasure:.4f}")
    print(f"Sensitivity (macro): {sensitivity:.4f}")
    print(f"Specificity (macro): {specificity:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"NPV (macro): {npv:.4f}")
    print(f"FPR (macro): {fpr:.4f}")
    print(f"FNR (macro): {fnr:.4f}")


        # Run RSA Optimization
    rsa = SelfAdaptiveRSA(
        n_dim=2,
        n_pop=10,
        lb=[1e-5, 0],
        ub=[1e-1, 0.1],
        max_iter=100
    )
    best_hyperparams, best_score = rsa.run()
    print("\nBest hyperparameters found by RSA:", best_hyperparams)
    print("Best fitness score:", best_score)


    # ----------------------------
    # Plot Confusion Matrix
    # ----------------------------
def plot_confusion_matrix2(cm, target_names, title='Confusion Matrix', cmap=None, normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Reds')

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18, fontweight='bold')
    plt.colorbar()

    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=0, fontsize=12, fontweight='bold')
    plt.yticks(tick_marks, target_names, fontsize=14, fontweight='bold')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i,
                 "{:0.4f}".format(cm[i, j]) if normalize else "{:,}".format(cm[i, j]),
                 horizontalalignment="center",
                 fontsize=12, fontweight='bold',
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Class', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=16, fontweight='bold')
    plt.show()


def confusion_matrix2_from_csv(csv_path):
    # Load the CSV
    df = pd.read_csv(csv_path)

    # Assume the CSV has 'Actual' and 'Predicted' columns
    y_true = df['class']
    y_pred = df['class']

    # Get the list of unique classes
    classes = sorted(list(set(y_true) | set(y_pred)))

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Plot the confusion matrix
    plot_confusion_matrix2(cm, target_names=classes, normalize=False, title="Confusion Matrix")

# Example usage:
csv_path = "/content/drive/MyDrive/Colab Notebooks/Plant Dieases/Test/concadenate fused_file1.csv"  # Replace with your actual path
confusion_matrix2_from_csv(csv_path)