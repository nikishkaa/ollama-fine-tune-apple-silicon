from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, \
   DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
import torch
from to_hf_dataset import prepare_dataset




# Подготовка данных
dataset = prepare_dataset()
dataset.save_to_disk("hf_dataset")


# Загрузка модели (например, Mistral)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
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
def tokenize_fn(examples):
   return tokenizer(examples["text"], truncation=True, max_length=512)


dataset = dataset.map(tokenize_fn, batched=True)


# Параметры обучения
training_args = TrainingArguments(
   output_dir="./results",
   num_train_epochs=3,
   per_device_train_batch_size=2,
   learning_rate=1e-5,
   fp16=True,
   logging_steps=10,
   save_strategy="no"
)


# Запуск обучения
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # Должно показать ~0.1% параметров


trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=dataset,
   data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)


trainer.train()
model.save_pretrained("./finetuned_model")