import pandas as pd
import json

def convert_csv_to_jsonl(csv_path, output_path):
    df = pd.read_csv(csv_path, sep=';', quotechar='"', encoding='utf-8')

    required_columns = {'name', 'description', 'precondition', 'scenario', 'expected_result'}
    if missing := required_columns - set(df.columns):
        raise ValueError(f"Отсутствуют колонки: {missing}")

    def safe_get(row, col, default=''):
        return row[col] if pd.notna(row[col]) else default

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            prompt = f"Создай тест-кейс для: {safe_get(row, 'description')}"

            completion = "\n".join([
                f"Название: {safe_get(row, 'name', 'Без названия')}",
                f"Предусловия: {safe_get(row, 'precondition')}",
                f"Сценарий: {safe_get(row, 'scenario')}",
                f"Ожидаемый результат: {safe_get(row, 'expected_result')}"
            ])

            entry = {
                "text": f"<s>[INST] {prompt} [/INST] {completion} </s>"
            }

            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    convert_csv_to_jsonl("your_data.csv", "ollama_dataset.jsonl") 