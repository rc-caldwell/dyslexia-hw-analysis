import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import multiprocessing

class HandwritingDataset(Dataset):
    def __init__(self, root_dir, split='Train', label='Normal'):
        self.paths = []
        for p in Path(root_dir).rglob("*.*"):
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                parts_lower = [part.lower() for part in p.parts]
                if split.lower() in parts_lower and label.lower() in parts_lower:
                    self.paths.append(p)
                    
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        return self.transform(img)

class ConstrainedAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        # Encoder: 64x64 -> 32x32 -> 16x16 -> 8x8
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU()
        )
        
        # The Choke Point
        self.flatten = nn.Flatten()
        self.encoder_linear = nn.Linear(64 * 8 * 8, latent_dim)
        
        # Decoder Expansion
        self.decoder_linear = nn.Linear(latent_dim, 64 * 8 * 8)
        self.unflatten = nn.Unflatten(1, (64, 8, 8))
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder_conv(x)
        x = self.flatten(x)
        latent = self.encoder_linear(x)
        x = self.decoder_linear(latent)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Constrained AE on {device}...")
    
    dataset = HandwritingDataset("data/raw", split="train", label="normal")
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=multiprocessing.cpu_count())
    
    model = ConstrainedAutoencoder(latent_dim=32).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 10
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch) 
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.6f}")
        
    torch.save(model.state_dict(), "data/processed/constrained_ae.pth")
    print("Saved to data/processed/constrained_ae.pth")