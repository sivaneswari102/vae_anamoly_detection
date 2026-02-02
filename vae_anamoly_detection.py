import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score

# -----------------------------
# Dataset Generation
# -----------------------------
def generate_data(n_steps=10000, n_features=15, anomaly_ratio=0.01):
    data = np.random.normal(0, 1, (n_steps, n_features))
    labels = np.zeros(n_steps)

    anomaly_indices = np.random.choice(
        n_steps, int(n_steps * anomaly_ratio), replace=False
    )

    for idx in anomaly_indices:
        data[idx] += np.random.normal(3, 0.5, n_features)
        labels[idx] = 1

    return data.astype(np.float32), labels.astype(int)

# -----------------------------
# VAE Model
# -----------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=5):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.mu = nn.Linear(32, latent_dim)
        self.logvar = nn.Linear(32, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        enc = self.encoder(x)
        mu = self.mu(enc)
        logvar = self.logvar(enc)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

# -----------------------------
# Loss Function (Beta-VAE)
# -----------------------------
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div, recon_loss

# -----------------------------
# Training
# -----------------------------
def train_vae(model, data, epochs=30, lr=1e-3, beta=1.0):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss, _ = vae_loss(recon, data, mu, logvar, beta)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# -----------------------------
# Anomaly Detection
# -----------------------------
def detect_anomalies(model, data, threshold):
    model.eval()
    with torch.no_grad():
        recon, _, _ = model(data)
        errors = torch.mean((recon - data) ** 2, dim=1)
    return errors.cpu().numpy() > threshold, errors.cpu().numpy()

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    X, y = generate_data()
    X_tensor = torch.tensor(X)

    normal_data = X_tensor[y == 0]

    model = VAE(input_dim=X.shape[1])
    train_vae(model, normal_data, beta=1.5)

    preds, errors = detect_anomalies(model, X_tensor, np.percentile(errors := errors if False else [], 99))
    
    # Proper threshold
    threshold = np.percentile(errors := detect_anomalies(model, normal_data, 0)[1], 99)

    y_pred, errors = detect_anomalies(model, X_tensor, threshold)

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print("\nFinal Performance Metrics")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
