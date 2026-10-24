import torch
from torch.utils.data import DataLoader

from evaluate import flow_log_prob, vae_negative_elbo


def train_vae(model, train_data, test_data, epochs=100, batch_size=256, seed=42):
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = {"train": [], "test": []}
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.loss(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        train_loss = total_loss / len(train_data)
        test_loss = vae_negative_elbo(model, test_data, batch_size)
        history["train"].append(train_loss)
        history["test"].append(test_loss)
        if epoch % 10 == 0:
            print(f"VAE     epoch {epoch:3d}/{epochs}: train negative ELBO {train_loss:.4f}, test {test_loss:.4f}")
    return history


def train_realnvp(model, train_data, test_data, epochs=100, batch_size=256, seed=42):
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = {"train": [], "test": []}
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.loss(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        train_loss = total_loss / len(train_data)
        test_loss = -flow_log_prob(model, test_data, batch_size).mean().item()
        history["train"].append(train_loss)
        history["test"].append(test_loss)
        if epoch % 10 == 0:
            print(f"RealNVP epoch {epoch:3d}/{epochs}: train exact NLL {train_loss:.4f}, test {test_loss:.4f}")
    return history
