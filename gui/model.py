import numpy as np
from astropy.io import fits
import cv2 as cv
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import distance_transform_edt

class Model:
    def __init__(self):
        self.fits_path = None
        self.kernel_size = 21
        self.threshold = 0.5
        self.blur_sigma = 5
        self.mask_dilate_size = 5
        self.attenuation_factor = 0.4
        self.multiscale_erosion = True  # Active l'érosion adaptative

    def load_fits(self, path):
        self.fits_path = path
        hdul = fits.open(path)
        data = hdul[0].data
        hdul.close()
        return data

    def process_image(self, data, kernel_size, threshold, blur_sigma, mask_dilate_size, attenuation_factor):
        # Handle both monochrome and color images
        if data.ndim == 3:
            if data.shape[0] == 3:
                data = np.transpose(data, (1, 2, 0))
            data_gray = np.mean(data, axis=2)
            image = np.zeros_like(data, dtype='uint8')
            for i in range(data.shape[2]):
                channel = data[:, :, i]
                image[:, :, i] = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype('uint8')
        else:
            data_gray = data
            image = ((data - data.min()) / (data.max() - data.min()) * 255).astype('uint8')

        # Star detection
        mean, median, std = sigma_clipped_stats(data_gray, sigma=3.0)
        daofind = DAOStarFinder(fwhm=5.0, threshold=median + threshold * std)
        sources = daofind(data_gray)

        mask = np.zeros_like(data_gray, dtype=np.uint8)
        if sources is not None:
            positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
            for pos in positions:
                x, y = int(pos[0]), int(pos[1])
                if 0 <= x < data_gray.shape[1] and 0 <= y < data_gray.shape[0]:
                    mask[y, x] = 255

        thresh_value = median + 3 * std
        bright_mask = (data_gray > thresh_value).astype(np.uint8) * 255
        mask = cv.bitwise_or(mask, bright_mask)

        # === ÉROSION ADAPTATIVE PAR ÉTOILE ===
        if self.multiscale_erosion and sources is not None:
            Ifinal = self._adaptive_erosion_per_star(
                image, data_gray, sources, mask, 
                kernel_size, blur_sigma, mask_dilate_size
            )
        else:
            # Érosion classique (ancienne méthode)
            kernel_dilate = np.ones((mask_dilate_size, mask_dilate_size), np.uint8)
            mask = cv.dilate(mask, kernel_dilate, iterations=2)
            
            kernel_erosion = np.ones((5, 5), np.uint8)
            Ierode = cv.erode(image, kernel_erosion, iterations=2)
            
            M = mask.astype(np.float32) / 255.0
            M = cv.GaussianBlur(M, (kernel_size, kernel_size), blur_sigma)
            
            if image.ndim == 3:
                M = np.stack([M] * image.shape[2], axis=2)
            
            Ifinal = (M * Ierode.astype(np.float32) + (1 - M) * image.astype(np.float32)).astype(np.uint8)

        # Reintegrate bright stars
        final = Ifinal.copy()
        if sources is not None:
            threshold_flux = np.percentile(sources['flux'], 90)
            for source in sources:
                if source['flux'] > threshold_flux:
                    x, y = int(source['xcentroid']), int(source['ycentroid'])
                    half_size = 15
                    x1, x2 = max(0, x - half_size), min(image.shape[1], x + half_size)
                    y1, y2 = max(0, y - half_size), min(image.shape[0], y + half_size)
                    star_patch = image[y1:y2, x1:x2].astype(np.float32)
                    attenuated_star = star_patch * attenuation_factor
                    final[y1:y2, x1:x2] = np.maximum(final[y1:y2, x1:x2], attenuated_star.astype(np.uint8))

        return final

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