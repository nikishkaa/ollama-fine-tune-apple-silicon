from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, \
   DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk
import torch
from to_hf_dataset import prepare_dataset

# Проверка доступности GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Подготовка данных
dataset = prepare_dataset()
dataset.save_to_disk("hf_dataset")

# Загрузка модели с явным указанием устройства
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    device_map=device,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    load_in_8bit=True if device == "cuda" else False
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.pad_token = tokenizer.eos_token

# Подготовка модели для обучения
if device == "cuda":
    model = prepare_model_for_kbit_training(model)
else:
    print("Warning: Running on CPU, training will be slower")

# Настройка LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Токенизация
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

dataset = dataset.map(tokenize_fn, batched=True)

# Параметры обучения
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=1e-5,
    fp16=device == "cuda",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2
)

# Запуск обучения
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()

# Сохранение модели и токенизатора
model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")