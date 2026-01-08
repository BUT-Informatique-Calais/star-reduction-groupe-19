from PyQt5.QtWidgets import QFileDialog, QMessageBox
import numpy as np
class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.data = None
        self.original_image = None
        self.processed_image = None
        self.showing_original = False


        self.view.load_button.clicked.connect(self.load_fits)
        self.view.kernel_slider.valueChanged.connect(self.update_kernel)
        self.view.threshold_slider.valueChanged.connect(self.update_threshold)
        self.view.blur_sigma_slider.valueChanged.connect(self.update_blur_sigma)
        self.view.mask_dilate_slider.valueChanged.connect(self.update_mask_dilate)
        self.view.attenuation_slider.valueChanged.connect(self.update_attenuation)
        self.view.toggle_button.clicked.connect(self.toggle_image)


    def load_fits(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self.view, "Charger fichier FITS", "", "FITS Files (*.fits);;All Files (*)", options=options)
        if file_path:
            try:
                self.data = self.model.load_fits(file_path)
                self.process_and_display()
            except Exception as e:
                QMessageBox.critical(self.view, "Erreur", f"Erreur lors du chargement du fichier FITS : {str(e)}")

    def update_kernel(self, value):
        if value % 2 == 0:
            value += 1
        self.model.kernel_size = value
        self.view.update_labels(self.model.kernel_size, self.model.threshold, self.model.blur_sigma, self.model.mask_dilate_size, self.model.attenuation_factor)
        if self.data is not None:
            self.process_and_display()

    def update_threshold(self, value):
        self.model.threshold = value / 10.0
        self.view.update_labels(self.model.kernel_size, self.model.threshold, self.model.blur_sigma, self.model.mask_dilate_size, self.model.attenuation_factor)
        if self.data is not None:
            self.process_and_display()

    def update_blur_sigma(self, value):
        self.model.blur_sigma = value
        self.view.update_labels(self.model.kernel_size, self.model.threshold, self.model.blur_sigma, self.model.mask_dilate_size, self.model.attenuation_factor)
        if self.data is not None:
            self.process_and_display()

    def update_mask_dilate(self, value):
        self.model.mask_dilate_size = value
        self.view.update_labels(self.model.kernel_size, self.model.threshold, self.model.blur_sigma, self.model.mask_dilate_size, self.model.attenuation_factor)
        if self.data is not None:
            self.process_and_display()

    def update_attenuation(self, value):
        self.model.attenuation_factor = value / 10.0
        self.view.update_labels(self.model.kernel_size, self.model.threshold, self.model.blur_sigma, self.model.mask_dilate_size, self.model.attenuation_factor)
        if self.data is not None:
            self.process_and_display()

    def process_and_display(self):
        try:
            # Gérer les images couleur et monochrome
            if self.data.ndim == 3:
                if self.data.shape[0] == 3:
                    data = np.transpose(self.data, (1, 2, 0))
                else:
                    data = self.data
                img = np.mean(data, axis=2)
            else:
                img = self.data

            self.original_image = ((img - img.min()) / (img.max() - img.min()) * 255).astype('uint8')

            # Image traitée
            self.processed_image = self.model.process_image(
                self.data,
                self.model.kernel_size,
                self.model.threshold,
                self.model.blur_sigma,
                self.model.mask_dilate_size,
                self.model.attenuation_factor
            )

            # Afficher l'image traitée par défaut
            self.showing_original = False
            self.view.update_image(self.processed_image)
            self.view.toggle_button.setText("Afficher image originale")

        except Exception as e:
            QMessageBox.critical(self.view, "Erreur", f"Erreur lors du traitement de l'image : {str(e)}")

    def toggle_image(self):
        if self.original_image is None or self.processed_image is None:
            return

        if self.showing_original:
            # Afficher l'image traitée
            self.view.update_image(self.processed_image)
            self.view.toggle_button.setText("Afficher image originale")
            self.showing_original = False
        else:
            # Afficher l'image originale
            self.view.update_image(self.original_image)
            self.view.toggle_button.setText("Afficher image traitée")
            self.showing_original = True
