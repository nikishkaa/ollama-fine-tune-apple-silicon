from datasets import Dataset
import json

def convert_to_hf_dataset(input_file, output_dir="hf_dataset"):
    # Читаем входной файл
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Преобразуем данные в формат для датасета
    texts = []
    for item in data:
        # Форматируем текст в стиле Mistral
        text = f"<s>[INST] {item['instruction']} [/INST] {item['output']}</s>"
        texts.append({"text": text})
    
    # Создаем датасет
    dataset = Dataset.from_dict({"text": [item["text"] for item in texts]})
    
    # Сохраняем датасет
    dataset.save_to_disk(output_dir)

if __name__ == "__main__":
    convert_to_hf_dataset("training_data.json") 