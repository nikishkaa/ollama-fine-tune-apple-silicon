from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
import torch
from to_hf_dataset import prepare_dataset

def train_model():
    # Подготовка данных
    dataset = prepare_dataset("your_data.csv")
    dataset.save_to_disk("hf_dataset")

    # Определение устройства
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Загрузка модели
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer.pad_token = tokenizer.eos_token

    # Настройка LoRA
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Токенизация
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            max_length=512,
            padding="max_length",
            truncation=True,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    # Параметры обучения
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Уменьшаем размер батча
        learning_rate=1e-5,
        fp16=False,
        logging_steps=10,
        save_strategy="no"
    )

    # Запуск обучения
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()
    model.save_pretrained("./finetuned_model")

if __name__ == "__main__":
    train_model() 