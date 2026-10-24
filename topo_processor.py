"""
topo_processor.py
Обработка топографических съёмок: распознавание горизонталей, 
построение ЦМР, расчёт выравнивания.
"""

import cv2
import numpy as np
import fitz  # PyMuPDF
from scipy import ndimage
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, List, Dict


class TopoProcessor:
    """Обработчик топографических съёмок."""
    
    def __init__(self):
        self.scale_factor = None  # метров на пиксель
        self.contours = []
        self.height_points = []
        self.dem = None  # Digital Elevation Model
        
    def load_file(self, file_path: str) -> np.ndarray:
        """Загрузка PDF или изображения."""
        if file_path.lower().endswith('.pdf'):
            return self._load_pdf(file_path)
        else:
            return self._load_image(file_path)
    
    def _load_pdf(self, pdf_path: str) -> np.ndarray:
        """Конвертация PDF в изображение."""
        doc = fitz.open(pdf_path)
        page = doc[0]  # Первая страница
        
        # Рендерим с высоким DPI для качества
        mat = fitz.Matrix(300/72, 300/72)  # 300 DPI
        pix = page.get_pixmap(matrix=mat)
        
        # Конвертируем в numpy array
        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.height, pix.width, pix.n)
        
        # Если RGBA, конвертируем в RGB
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        doc.close()
        return img
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """Загрузка изображения."""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {img_path}")
        return img
    
    def set_scale(self, scale_meters_per_pixel: float):
        """Установка масштаба (метров на пиксель)."""
        self.scale_factor = scale_meters_per_pixel
        print(f"Масштаб установлен: {scale_meters_per_pixel:.4f} м/пикс")
    
    def detect_contours(self, img: np.ndarray) -> np.ndarray:
        """Обнаружение горизонталей (изогипс)."""
        # Конвертируем в grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Бинаризация (инвертируем, чтобы линии были белыми)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Убираем шум
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Находим контуры
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        # Фильтруем по размеру (убираем слишком маленькие)
        self.contours = [c for c in contours if cv2.contourArea(c) > 100]
        
        print(f"Обнаружено горизонталей: {len(self.contours)}")
        return binary
    
    def extract_height_points(self, binary_img: np.ndarray, 
                             contour_heights: List[float] = None) -> Dict:
        """
        Извлечение точек с высотами.
        contour_heights: список высот для каждой горизонтали (если известен).
        """
        # Если высоты не заданы, присваиваем условные
        if contour_heights is None:
            contour_heights = np.linspace(100, 200, len(self.contours))
        
        height_points = []
        
        for i, contour in enumerate(self.contours):
            height = contour_heights[i % len(contour_heights)]
            
            # Берём точки контура
            for point in contour[::10]:  # Каждую 10-ю точку для скорости
                x, y = point[0]
                height_points.append({
                    'x': x,
                    'y': y,
                    'height': height
                })
        
        self.height_points = height_points
        print(f"Извлечено точек с высотами: {len(height_points)}")
        return {'points': height_points}
    
    def build_dem(self, img_shape: Tuple[int, int]) -> np.ndarray:
        """Построение цифровой модели рельефа (интерполяция)."""
        if not self.height_points:
            raise ValueError("Сначала извлеките точки с высотами")
        
        # Создаём сетку
        y_coords = np.arange(img_shape[0])
        x_coords = np.arange(img_shape[1])
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Подготовка данных для интерполяции
        points = np.array([[p['x'], p['y']] for p in self.height_points])
        values = np.array([p['height'] for p in self.height_points])
        
        # Интерполяция (метод кубических сплайнов)
        print("Построение ЦМР (интерполяция)...")
        self.dem = griddata(points, values, (X, Y), method='cubic')
        
        # Заполняем NaN значения (границы)
        mask = np.isnan(self.dem)
        self.dem[mask] = griddata(points, values, (X[mask], Y[mask]), method='nearest')
        
        print(f"ЦМР построена: {self.dem.shape}")
        return self.dem
    
    def create_heatmap(self, dem: np.ndarray = None) -> np.ndarray:
        """Создание тепловой карты рельефа."""
        if dem is None:
            dem = self.dem
        
        if dem is None:
            raise ValueError("Сначала постройте ЦМР")
        
        # Нормализация для визуализации
        dem_norm = (dem - dem.min()) / (dem.max() - dem.min())
        
        # Применяем цветовую карту (jet: синий→красный)
        colormap = plt.cm.jet
        heatmap = colormap(dem_norm)
        
        # Конвертируем в RGB (0-255)
        heatmap_rgb = (heatmap[:, :, :3] * 255).astype(np.uint8)
        
        return heatmap_rgb
    
    def calculate_leveling(self, target_height: float = None) -> Dict:
        """
        Расчёт выравнивания рельефа.
        target_height: целевая высота (если None, используется средняя).
        """
        if self.dem is None:
            raise ValueError("Сначала постройте ЦМР")
        
        # Если целевая высота не задана, используем среднюю
        if target_height is None:
            target_height = np.mean(self.dem)
        
        # Разница высот
        height_diff = self.dem - target_height
        
        # Объём земляных работ (в условных единицах, если есть масштаб)
        if self.scale_factor:
            # Объём в кубических метрах (приближённо)
            pixel_area = self.scale_factor ** 2  # м² на пиксель
            volume = np.sum(height_diff) * pixel_area
        else:
            volume = np.sum(height_diff)
        
        # Статистика
        stats = {
            'target_height': target_height,
            'min_height': float(self.dem.min()),
            'max_height': float(self.dem.max()),
            'mean_height': float(self.dem.mean()),
            'volume_cut': float(np.sum(height_diff[height_diff > 0])),  # Срезка
            'volume_fill': float(np.abs(np.sum(height_diff[height_diff < 0]))),  # Насыпь
            'net_volume': float(volume),
            'height_diff_map': height_diff
        }
        
        print(f"Целевая высота: {target_height:.2f}")
        print(f"Объём срезки: {stats['volume_cut']:.2f}")
        print(f"Объём насыпи: {stats['volume_fill']:.2f}")
        print(f"Баланс: {stats['net_volume']:.2f}")
        
        return stats
    
    def create_leveling_visualization(self, stats: Dict) -> np.ndarray:
        """Визуализация выравнивания (красный=срезка, синий=насыпь)."""
        height_diff = stats['height_diff_map']
        
        # Нормализация
        max_diff = max(abs(height_diff.min()), abs(height_diff.max()))
        normalized = height_diff / max_diff
        
        # Цветовая карта: красный (срезка) → белый (0) → синий (насыпь)
        colormap = plt.cm.RdBu_r  # Red-Blue diverging
        viz = colormap((normalized + 1) / 2)  # Сдвиг в диапазон [0, 1]
        
        # Конвертируем в RGB
        viz_rgb = (viz[:, :, :3] * 255).astype(np.uint8)
        
        return viz_rgb


def process_topo(file_path: str, scale_meters_per_pixel: float = None,
                target_height: float = None) -> Dict:
    """
    Основная функция обработки топографической съёмки.
    
    Args:
        file_path: путь к PDF или изображению
        scale_meters_per_pixel: масштаб (метров на пиксель)
        target_height: целевая высота для выравнивания
    
    Returns:
        Словарь с результатами и визуализациями
    """
    processor = TopoProcessor()
    
    # 1. Загрузка
    print("Загрузка файла...")
    img = processor.load_file(file_path)
    print(f"Размер изображения: {img.shape}")
    
    # 2. Установка масштаба
    if scale_meters_per_pixel:
        processor.set_scale(scale_meters_per_pixel)
    
    # 3. Обнаружение горизонталей
    print("Обнаружение горизонталей...")
    binary = processor.detect_contours(img)
    
    # 4. Извлечение точек с высотами
    print("Извлечение точек с высотами...")
    processor.extract_height_points(binary)
    
    # 5. Построение ЦМР
    print("Построение цифровой модели рельефа...")
    dem = processor.build_dem(img.shape[:2])
    
    # 6. Тепловая карта
    print("Создание тепловой карты...")
    heatmap = processor.create_heatmap(dem)
    
    # 7. Расчёт выравнивания
    print("Расчёт выравнивания...")
    stats = processor.calculate_leveling(target_height)
    
    # 8. Визуализация выравнивания
    leveling_viz = processor.create_leveling_visualization(stats)
    
    return {
        'original_image': img,
        'binary_contours': binary,
        'heatmap': heatmap,
        'leveling_map': leveling_viz,
        'statistics': stats
    }