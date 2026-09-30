import torch
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

def heal_pruned_model(
    model,
    tokenizer,
    train_samples=256,
    max_length=256,
    batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    lora_r=8,
    lora_alpha=16
):
    """
    Прилага леко LoRA дообучаване (Healing) върху орязан модел за възстановяване на PPL,
    след което слива LoRA теглата обратно в базовия модел.
    """
    print(f"\n[Healing] 1. Конфигуриране на LoRA адаптери (r={lora_r}, alpha={lora_alpha})...")
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Закачане на LoRA
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    # 2. Подготовка на малък dataset за дообучаване
    print(f"[Healing] 2. Зареждане на данни за healing ({train_samples} примера)...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    
    # Филтрираме кратки или празни редове
    texts = [t for t in dataset["text"] if len(t.strip()) > 50][:train_samples]
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False
        )
    
    from datasets import Dataset
    train_dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    # 3. Настройки за обучение
    training_args = TrainingArguments(
        output_dir="./tmp_heal_output",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_steps=10,
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
        report_to="none"
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    print("[Healing] 3. Стартиране на адаптивното дообучаване...")
    trainer.train()
    
    # 4. Сливане на LoRA адаптерите обратно в модела (Merge & Unload)
    print("[Healing] 4. Сливане на LoRA теглата обратно в основния модел...")
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()
    
    return merged_model
