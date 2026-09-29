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
                    
        # Normalize and convert to tensors natively
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

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: 64x64 -> 32x32 -> 16x16 -> 8x8 latent space
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder: 8x8 -> 16x16 -> 32x32 -> 64x64 original dimensions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # Squashes output pixel values between [0, 1]
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

if __name__ == "__main__":
    # WSL2 can utilize CUDA if configured, otherwise falls back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading Normal baseline dataset...")
    dataset = HandwritingDataset("data/raw", split="train", label="normal")
    
    # Parallelize data loading
    cores = multiprocessing.cpu_count()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=cores)
    
    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 10
    print(f"Starting training on {len(dataset)} images for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch)
            
            # The label is the input image itself
            loss = criterion(outputs, batch) 
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        
    torch.save(model.state_dict(), "data/processed/autoencoder.pth")
    print("\nModel weights saved to data/processed/autoencoder.pth")