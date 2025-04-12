import pandas as pd
from datasets import Dataset


def prepare_dataset():
   df = pd.read_csv("your_data.csv", sep=';', encoding='utf-8')


   def generate_text(row):
       return f"""<s>[INST] Создай тест-кейс для: {row['description']} [/INST]
       Название: {row['name']}
       Предварительные условия: {row['precondition']}
       Сценарий: {row['scenario']}
       Ожидаемый результат: {row['expected_result']}</s>"""


   df['text'] = df.apply(generate_text, axis=1)
   return Dataset.from_pandas(df[['text']])