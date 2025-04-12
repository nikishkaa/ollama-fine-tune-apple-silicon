import pandas as pd
from datasets import Dataset

def prepare_dataset(csv_path):
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8')

    def generate_input_output(row):
        input_text = f"Создай рецепт для: {row['description']}"
        output_text = f"Название: {row['name']}\nПредварительные условия: {row['precondition']}\nСценарий: {row['scenario']}\nОжидаемый результат: {row['expected_result']}"
        return {
            "input_text": input_text,
            "output_text": output_text
        }

    processed_data = [generate_input_output(row) for _, row in df.iterrows()]
    return Dataset.from_pandas(pd.DataFrame(processed_data))

if __name__ == "__main__":
    dataset = prepare_dataset("your_data.csv")
    dataset.save_to_disk("hf_dataset") 