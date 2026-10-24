"""
app.py
Веб-интерфейс для анализа топографических съёмок.
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from topo_processor import process_topo


def analyze_topo(file, scale_input, target_height_input):
    """Обработка загруженного файла."""
    if file is None:
        return "Загрузите файл", None, None, None, None
    
    try:
        # Парсим масштаб
        scale = float(scale_input) if scale_input else None
        
        # Парсим целевую высоту
        target_height = float(target_height_input) if target_height_input else None
        
        # Обрабатываем
        results = process_topo(file.name, scale, target_height)
        
        # Формируем текст статистики
        stats = results['statistics']
        stats_text = f"""
📊 **Статистика рельефа:**

- Минимальная высота: {stats['min_height']:.2f} м
- Максимальная высота: {stats['max_height']:.2f} м
- Средняя высота: {stats['mean_height']:.2f} м
- Целевая высота: {stats['target_height']:.2f} м

🚜 **Земляные работы:**

- Объём срезки: {stats['volume_cut']:.2f} м³
- Объём насыпи: {stats['volume_fill']:.2f} м³
- Баланс: {stats['net_volume']:.2f} м³

💡 **Рекомендации:**
- {"Баланс срезки/насыпки оптимальный" if abs(stats['net_volume']) < stats['volume_cut'] * 0.1 else "Требуется корректировка целевой высоты"}
        """
        
        return (
            stats_text,
            results['original_image'],
            results['heatmap'],
            results['leveling_map'],
            "✅ Обработка завершена успешно!"
        )
    
    except Exception as e:
        return f"❌ Ошибка: {str(e)}", None, None, None, None


# Создаём интерфейс
with gr.Blocks(title="Анализ топографических съёмок") as demo:
    gr.Markdown("""
    # 🗺️ Анализатор топографических съёмок
    
    Загрузите PDF или изображение топографической съёмки для анализа рельефа.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Загрузите файл (PDF или изображение)", 
                                file_types=[".pdf", ".png", ".jpg", ".jpeg"])
            
            scale_input = gr.Number(
                label="Масштаб (метров на пиксель)",
                value=0.1,
                info="Например: 0.1 = 10 см на пиксель. Оставьте пустым для относительных расчётов."
            )
            
            target_height_input = gr.Number(
                label="Целевая высота для выравнивания (м)",
                value=None,
                info="Оставьте пустым для автоматического расчёта (средняя высота)"
            )
            
            analyze_btn = gr.Button("🔍 Анализировать", variant="primary")
        
        with gr.Column(scale=2):
            status_output = gr.Textbox(label="Статус", lines=1)
            
            with gr.Tabs():
                with gr.Tab("📊 Статистика"):
                    stats_output = gr.Markdown()
                
                with gr.Tab("🖼️ Оригинал"):
                    original_output = gr.Image(label="Исходное изображение")
                
                with gr.Tab("🌡️ Тепловая карта"):
                    heatmap_output = gr.Image(label="Тепловая карта рельефа")
                
                with gr.Tab("🚜 Выравнивание"):
                    leveling_output = gr.Image(label="Карта выравнивания (красный=срезка, синий=насыпь)")
    
    analyze_btn.click(
        fn=analyze_topo,
        inputs=[file_input, scale_input, target_height_input],
        outputs=[stats_output, original_output, heatmap_output, leveling_output, status_output]
    )
    
    gr.Markdown("""
    ---
    ### 📖 Инструкция:
    1. Загрузите PDF или изображение топографической съёмки
    2. Укажите масштаб (сколько метров в одном пикселе)
    3. Укажите целевую высоту для выравнивания (или оставьте пустым)
    4. Нажмите "Анализировать"
    5. Изучите результаты во вкладках
    
    ### 💡 Советы:
    - Для точного расчёта объёмов обязательно укажите масштаб
    - Красный цвет на карте выравнивания = нужно срезать грунт
    - Синий цвет = нужно насыпать грунт
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)