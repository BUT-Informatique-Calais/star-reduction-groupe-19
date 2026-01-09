import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
from astropy.io import fits
from model import TinyUNet, Model
import os
import cv2 as cv

class StarDataset(Dataset):
    def __init__(self, fits_dir):
        # Convertir en chemin absolu pour être sûr
        self.abs_path = os.path.abspath(fits_dir)
        print(f"--- Diagnostic du dossier ---")
        print(f"Recherche dans : {self.abs_path}")
        
        if not os.path.exists(self.abs_path):
            print(f"ERREUR : Le dossier n'existe pas !")
            self.files = []
        else:
            # Lister TOUS les fichiers pour voir ce qu'il y a dedans
            all_files = os.listdir(self.abs_path)
            print(f"Fichiers totaux trouvés dans le dossier : {len(all_files)}")
            
            # Filtrer par extension .fits
            self.files = [os.path.join(self.abs_path, f) for f in all_files 
                          if f.lower().endswith('.fits')]
            
            print(f"Fichiers .fits détectés : {len(self.files)}")
            if len(self.files) > 0:
                print(f"Exemple de fichier : {self.files[0]}")
        print(f"-----------------------------")
        
        self.base_model = Model()
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with fits.open(self.files[idx]) as hdul:
            data = hdul[0].data.astype(np.float32)
        
        # Gérer le cas des images couleur (3D) pour la détection
        if data.ndim == 3:
            if data.shape[0] == 3: # Format [C, H, W]
                data_for_detect = np.mean(data, axis=0)
            else: # Format [H, W, C]
                data_for_detect = np.mean(data, axis=2)
        else:
            data_for_detect = data

        # 1. Génération du Masque Label via ton algo actuel
        mask = self.base_model.detect_stars(data_for_detect)
        mask = (mask > 0).astype(np.float32)
        
        # 2. Préparation de l'entrée (Normalisation log)
        img_norm = np.log1p(data_for_detect - data_for_detect.min())
        img_norm = (img_norm / (img_norm.max() + 1e-8)).astype(np.float32)
        
        # On force une taille fixe (ex: 512x512) pour que le stack fonctionne
        target_size = (512, 512) 
        img_norm = cv.resize(img_norm, target_size, interpolation=cv.INTER_AREA)
        mask = cv.resize(mask, target_size, interpolation=cv.INTER_NEAREST)
        # --------------------------------------

        input_tensor = torch.from_numpy(img_norm).unsqueeze(0) # [1, 512, 512]
        target_tensor = torch.from_numpy(mask).unsqueeze(0)   # [1, 512, 512]
        
        return input_tensor, target_tensor

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Entraînement sur : {device}")

    model = TinyUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss() # Binary Cross Entropy pour masque 0/1

    dataset = StarDataset('gui/data') 
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model.train()
    for epoch in range(20):
        total_loss = 0
        for batch_in, batch_target in dataloader:
            batch_in, batch_target = batch_in.to(device), batch_target.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_in)
            loss = criterion(outputs, batch_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/20, Loss: {total_loss/len(dataloader):.4f}")

    # Sauvegarde finale
    torch.save(model.state_dict(), 'star_unet.pth')
    print("Modèle sauvegardé sous 'star_unet.pth'")

if __name__ == "__main__":
    train()