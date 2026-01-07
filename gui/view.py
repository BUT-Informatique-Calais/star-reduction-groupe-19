from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np

class View(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Selective Erosion GUI')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        # Button to load FITS
        self.load_button = QPushButton('Charger fichier FITS')
        layout.addWidget(self.load_button)

        # Sliders
        slider_layout = QHBoxLayout()

        self.kernel_label = QLabel('Taille du noyau: 21')
        slider_layout.addWidget(self.kernel_label)

        self.kernel_slider = QSlider(Qt.Horizontal)
        self.kernel_slider.setMinimum(5)
        self.kernel_slider.setMaximum(51)
        self.kernel_slider.setValue(21)
        self.kernel_slider.setTickInterval(2)
        self.kernel_slider.setTickPosition(QSlider.TicksBelow)
        slider_layout.addWidget(self.kernel_slider)

        self.threshold_label = QLabel('Seuil du masque: 0.5')
        slider_layout.addWidget(self.threshold_label)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(20)
        self.threshold_slider.setValue(5)  # 0.5 * 10
        self.threshold_slider.setTickInterval(1)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        slider_layout.addWidget(self.threshold_slider)

        self.blur_sigma_label = QLabel('Sigma du flou: 5')
        slider_layout.addWidget(self.blur_sigma_label)

        self.blur_sigma_slider = QSlider(Qt.Horizontal)
        self.blur_sigma_slider.setMinimum(1)
        self.blur_sigma_slider.setMaximum(20)
        self.blur_sigma_slider.setValue(5)
        self.blur_sigma_slider.setTickInterval(1)
        self.blur_sigma_slider.setTickPosition(QSlider.TicksBelow)
        slider_layout.addWidget(self.blur_sigma_slider)

        self.mask_dilate_label = QLabel('Taille dilatation: 5')
        slider_layout.addWidget(self.mask_dilate_label)

        self.mask_dilate_slider = QSlider(Qt.Horizontal)
        self.mask_dilate_slider.setMinimum(1)
        self.mask_dilate_slider.setMaximum(15)
        self.mask_dilate_slider.setValue(5)
        self.mask_dilate_slider.setTickInterval(1)
        self.mask_dilate_slider.setTickPosition(QSlider.TicksBelow)
        slider_layout.addWidget(self.mask_dilate_slider)

        self.attenuation_label = QLabel('Atténuation: 0.4')
        slider_layout.addWidget(self.attenuation_label)

        self.attenuation_slider = QSlider(Qt.Horizontal)
        self.attenuation_slider.setMinimum(0)
        self.attenuation_slider.setMaximum(10)
        self.attenuation_slider.setValue(4)  # 0.4 * 10
        self.attenuation_slider.setTickInterval(1)
        self.attenuation_slider.setTickPosition(QSlider.TicksBelow)
        slider_layout.addWidget(self.attenuation_slider)

        layout.addLayout(slider_layout)

        # Image display
        self.image_label = QLabel('Image will appear here')
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        self.setLayout(layout)

    def update_image(self, image):
        image = np.ascontiguousarray(image)  # Ensure contiguous array
        if image.ndim == 3:
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_img = QImage(image.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            height, width = image.shape
            bytes_per_line = width
            q_img = QImage(image.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def update_labels(self, kernel_size, threshold, blur_sigma, mask_dilate_size, attenuation_factor):
        self.kernel_label.setText(f'Taille du noyau: {kernel_size}')
        self.threshold_label.setText(f'Seuil du masque: {threshold:.1f}')
        self.blur_sigma_label.setText(f'Sigma du flou: {blur_sigma}')
        self.mask_dilate_label.setText(f'Taille dilatation: {mask_dilate_size}')
        self.attenuation_label.setText(f'Atténuation: {attenuation_factor:.1f}')