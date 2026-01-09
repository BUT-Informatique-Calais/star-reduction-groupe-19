import numpy as np
from astropy.io import fits
import cv2 as cv
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn as nn

class Model:
    def __init__(self):
        self.fits_path = None
        self.kernel_size = 21
        self.threshold = 0.5
        self.blur_sigma = 5
        self.mask_dilate_size = 5
        self.attenuation_factor = 0.4
        self.multiscale_erosion = True  # Active l'érosion adaptative
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ai_model = TinyUNet().to(self.device)
        self.ai_loaded = False
        self.use_ai = False
        
        # Charger les poids si le fichier existe
        try:
            self.ai_model.load_state_dict(torch.load('star_unet.pth', map_location=self.device))
            self.ai_model.eval()
            self.ai_loaded = True
        except:
            print("Mode IA : Aucun modèle pré-entraîné trouvé (star_unet.pth)")

    def load_fits(self, path):
        self.fits_path = path
        hdul = fits.open(path)
        data = hdul[0].data
        hdul.close()
        return data

    def predict_star_mask(self, data):
        """Utilise l'IA pour générer le masque"""
        if not self.ai_loaded:
            return None
            
        # Normalisation identique à l'entraînement
        img_norm = np.log1p(data - data.min())
        img_norm = (img_norm / (img_norm.max() + 1e-8)).astype(np.float32)
        
        h_orig, w_orig = data.shape[:2]
        img_resized = cv.resize(img_norm, (512, 512))
        
        input_t = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.ai_model(input_t)
        
        mask = output.cpu().numpy()[0, 0]
        mask_resized = cv.resize(mask, (w_orig, h_orig))
        return (mask_resized > 0.5).astype(np.uint8) * 255

    def detect_stars(self, data):
        """Génère un masque binaire des étoiles (le label pour l'IA)"""
        # Conversion en gris si nécessaire
        if data.ndim == 3:
            data_gray = np.mean(data, axis=2)
        else:
            data_gray = data

        # Stats pour le seuillage (comme dans ton code actuel)
        mean, median, std = sigma_clipped_stats(data_gray, sigma=3.0)
        
        # Utilisation de DAOStarFinder (ta méthode actuelle)
        daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
        sources = daofind(data_gray - median)
        
        # Création du masque vide
        mask = np.zeros(data_gray.shape, dtype=np.uint8)
        
        if sources is not None:
            for x, y in zip(sources['xcentroid'], sources['ycentroid']):
                cv.circle(mask, (int(x), int(y)), int(self.mask_dilate_size), 1, -1)
        
        return mask

    def process_image(self, data, kernel_size, threshold, blur_sigma, mask_dilate_size, attenuation_factor):
        """Fonction principale de traitement avec gestion IA/Classique"""
        try:
            # 1. Conversion des données brutes en image 8-bit pour le traitement
            # On utilise une normalisation robuste pour éviter les images noires
            data_min, data_max = data.min(), data.max()
            img_8bit = ((data - data_min) / (data_max - data_min + 1e-8) * 255).astype('uint8')

            # 2. Préparation du masque d'étoiles
            if self.use_ai and self.ai_loaded:
                # Utilisation de l'IA (U-Net)
                star_mask_global = self.predict_star_mask(data)
            else:
                # Algorithme classique (DAOStarFinder)
                if data.ndim == 3:
                    data_gray = np.mean(data, axis=2)
                else:
                    data_gray = data
                
                mean, median, std = sigma_clipped_stats(data_gray, sigma=3.0)
                daofind = DAOStarFinder(fwhm=3.0, threshold=threshold * std)
                sources = daofind(data_gray - median)
                
                star_mask_global = np.zeros(data_gray.shape, dtype=np.uint8)
                if sources is not None:
                    for x, y in zip(sources['xcentroid'], sources['ycentroid']):
                        cv.circle(star_mask_global, (int(x), int(y)), int(mask_dilate_size), 255, -1)

            # 3. Application de l'érosion (Multiscale ou Simple)
            if self.multiscale_erosion:
                # On utilise img_8bit pour l'érosion, pas data
                return self._apply_multiscale_erosion(img_8bit, star_mask_global, kernel_size, attenuation_factor, blur_sigma)
            else:
                # Version simple
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                eroded = cv.erode(img_8bit, kernel)
                
                mask_norm = cv.GaussianBlur(star_mask_global.astype(float) / 255.0, (kernel_size | 1, kernel_size | 1), blur_sigma)
                
                if img_8bit.ndim == 3 and mask_norm.ndim == 2:
                    mask_norm = np.stack([mask_norm] * 3, axis=2)
                
                final_image = (img_8bit * (1 - mask_norm * attenuation_factor) + eroded * (mask_norm * attenuation_factor))
                return final_image.astype('uint8')

        except Exception as e:
            print(f"Erreur lors du traitement de l'image : {str(e)}")
            # En cas d'erreur, on retourne au moins l'image originale convertie
            return ((data - data.min()) / (data.max() - data.min() + 1e-8) * 255).astype('uint8')

    def _adaptive_erosion_per_star(self, image, data_gray, sources, mask, kernel_size, blur_sigma, mask_dilate_size):
        """
        Applique une érosion adaptative individuellement pour chaque étoile.
        Le niveau d'érosion est proportionnel au flux et au FWHM de l'étoile.
        """
        h, w = data_gray.shape
        
        # Normaliser les flux pour déterminer les niveaux d'érosion
        flux_values = sources['flux']
        flux_min, flux_max = flux_values.min(), flux_values.max()
        
        # Créer une carte d'érosion (carte qui indique quel niveau d'érosion appliquer)
        erosion_map = np.zeros((h, w), dtype=np.float32)
        star_mask_global = np.zeros((h, w), dtype=np.uint8)
        
        # Pour chaque étoile, définir sa zone d'influence et son niveau d'érosion
        for source in sources:
            x, y = int(source['xcentroid']), int(source['ycentroid'])
            if not (0 <= x < w and 0 <= y < h):
                continue
                
            flux = source['flux']
            fwhm = source.get('fwhm', 5.0)  # Utiliser FWHM si disponible
            
            # Calculer le niveau d'érosion basé sur le flux (normalisé entre 0 et 1)
            if flux_max > flux_min:
                flux_normalized = (flux - flux_min) / (flux_max - flux_min)
            else:
                flux_normalized = 0.5
            
            # Rayon d'influence proportionnel au FWHM et au flux
            # Plus l'étoile est brillante, plus le rayon est grand
            influence_radius = int(fwhm * 2.5 + flux_normalized * 15)
            influence_radius = max(8, min(influence_radius, 40))  # Limiter entre 8 et 40 pixels
            
            # Niveau d'érosion : 0 = pas d'érosion, 1 = érosion maximale
            # Les étoiles brillantes ont plus d'érosion (courbe exponentielle modérée)
            erosion_level = flux_normalized ** 0.8  # Moins agressif pour éviter les trous noirs
            
            # Créer un masque circulaire pour cette étoile avec gradient
            y_grid, x_grid = np.ogrid[-y:h-y, -x:w-x]
            distance_from_star = np.sqrt(x_grid*x_grid + y_grid*y_grid)
            
            # Masque avec gradient radial (1 au centre, 0 au bord)
            star_influence = np.maximum(0, 1 - distance_from_star / influence_radius)
            
            # Mettre à jour la carte d'érosion (prendre le max pour les zones qui se chevauchent)
            erosion_map = np.maximum(erosion_map, star_influence * erosion_level)
            
            # Marquer la zone d'influence de l'étoile
            star_mask_global = np.maximum(star_mask_global, (star_influence > 0.1).astype(np.uint8) * 255)
        
        # Appliquer plusieurs niveaux d'érosion avec préservation des détails
        # Créer 5 images érodées avec différentes intensités (moins agressif)
        erosion_levels = []
        for i in range(5):
            kernel_size_erosion = 3 + i * 2  # 3, 5, 7, 9, 11
            iterations = 1 + i  # 1, 2, 3, 4, 5
            kernel = np.ones((kernel_size_erosion, kernel_size_erosion), np.uint8)
            eroded = cv.erode(image, kernel, iterations=iterations)
            
            # Limiter l'érosion pour éviter les trous noirs complets
            # On garde au minimum 10% de la valeur originale
            eroded = np.maximum(eroded, (image * 0.1).astype(np.uint8))
            
            erosion_levels.append(eroded.astype(np.float32))
        
        # Interpoler entre les niveaux d'érosion selon la carte d'érosion
        # erosion_map va de 0 (pas d'érosion) à 1 (érosion maximale)
        result = np.zeros_like(image, dtype=np.float32)
        
        # Diviser erosion_map en 4 segments pour interpoler entre les 5 niveaux
        for i in range(4):
            lower_bound = i / 4.0
            upper_bound = (i + 1) / 4.0
            
            # Trouver les pixels dans cette plage
            mask_segment = ((erosion_map >= lower_bound) & (erosion_map < upper_bound))
            
            if np.any(mask_segment):
                # Interpolation linéaire entre niveau i et i+1
                alpha = (erosion_map - lower_bound) / (upper_bound - lower_bound)
                alpha = np.clip(alpha, 0, 1)
                
                if image.ndim == 3:
                    alpha_3d = np.stack([alpha] * image.shape[2], axis=2)
                    mask_segment_3d = np.stack([mask_segment] * image.shape[2], axis=2)
                    interpolated = (1 - alpha_3d) * erosion_levels[i] + alpha_3d * erosion_levels[i+1]
                    result = np.where(mask_segment_3d, interpolated, result)
                else:
                    interpolated = (1 - alpha) * erosion_levels[i] + alpha * erosion_levels[i+1]
                    result = np.where(mask_segment, interpolated, result)
        
        # Gérer les valeurs maximales (erosion_map == 1) - moins agressif
        mask_max = (erosion_map >= 0.75)
        if np.any(mask_max):
            if image.ndim == 3:
                mask_max_3d = np.stack([mask_max] * image.shape[2], axis=2)
                result = np.where(mask_max_3d, erosion_levels[4], result)
            else:
                result = np.where(mask_max, erosion_levels[4], result)
        
        # Flouter la carte d'érosion pour des transitions douces
        erosion_map_smooth = cv.GaussianBlur(erosion_map, (kernel_size, kernel_size), blur_sigma)
        
        if image.ndim == 3:
            erosion_map_smooth = np.stack([erosion_map_smooth] * image.shape[2], axis=2)
        
        # Mélanger avec l'image originale dans les zones sans étoiles
        star_mask_smooth = cv.GaussianBlur(star_mask_global.astype(np.float32) / 255.0, 
                                           (kernel_size, kernel_size), blur_sigma)
        
        if image.ndim == 3:
            star_mask_smooth = np.stack([star_mask_smooth] * image.shape[2], axis=2)
        
        # Résultat final : mélange entre image érodée et originale
        final_result = (
            star_mask_smooth * result +
            (1 - star_mask_smooth) * image.astype(np.float32)
        ).astype(np.uint8)
        
        return final_result
    
    def segment_stars_ai(self, image_data):
        """ Utilise l'IA pour créer un masque d'étoiles précis """
        # 1. Préparation (Normalisation log + conversion Tensor)
        img_norm = np.log1p(image_data - image_data.min())
        img_norm = (img_norm / img_norm.max()).astype(np.float32)
        
        tensor_in = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # 2. Inférence
        with torch.no_grad():
            mask_pred = self.ai_model(tensor_in)
        
        mask_np = mask_pred.cpu().numpy()[0, 0]
        return mask_np > 0.5 # Retourne un masque binaire

    def process_image_ai(self, data, attenuation):
        """ Nouvelle méthode de traitement utilisant le masque IA """
        mask = self.segment_stars_ai(data)
        
        # Au lieu d'éroder toute l'image, on ne traite que là où l'IA a vu des étoiles
        star_layer = data * mask
        sky_background = data * (~mask)
        
        # Réduction douce des étoiles
        reduced_stars = star_layer * (1 - attenuation)
        
        return sky_background + reduced_stars
    

class TinyUNet(nn.Module):
    def __init__(self):
        super(TinyUNet, self).__init__()
        
        # Encodeur (Descente)
        self.enc1 = self.conv_block(1, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(32, 64)
        
        # Décodeur (Montée)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(64, 32) # 64 car concaténation
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(32, 16)
        
        # Sortie (Masque binaire)
        self.final = nn.Conv2d(16, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.sigmoid(self.final(d1))