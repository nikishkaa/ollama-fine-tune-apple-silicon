import pandas as pd
from datasets import Dataset

def prepare_dataset(csv_path):
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8')

    def generate_text(row):
        return f"""<s>[INST] Создай рецепт для: {row['description']} [/INST]
Название: {row['name']}
Предварительные условия: {row['precondition']}
Сценарий: {row['scenario']}
Ожидаемый результат: {row['expected_result']}</s>"""

    df['text'] = df.apply(generate_text, axis=1)
    return Dataset.from_pandas(df[['text']])

if __name__ == "__main__":
    dataset = prepare_dataset("your_data.csv")
    dataset.save_to_disk("hf_dataset") 