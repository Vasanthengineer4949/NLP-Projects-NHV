# pip install accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.30.2 trl==0.4.7

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

def finetune_llama_v2():
    data = load_dataset("timdettmers/openassistant-guanaco", split="train")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf", quantization_config=bnb_config, device_map={"": 0}
    )
    model.config.use_cache=False
    model.config.pretraining_tp=1
    peft_config = LoraConfig(
        r=64, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    training_arguments = TrainingArguments(
        output_dir="llama2_finetuned_chatbot",
        per_device_train_batch_size=12,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=1,
        max_steps=20,
        fp16=True,
        push_to_hub=True
    )
    trainer = SFTTrainer(
        model=model, 
        train_dataset=data, 
        peft_config=peft_config, 
        dataset_text_field="text", 
        training_arguments=training_arguments, 
        tokenizer=tokenizer, 
        packing=False
    )
    trainer.train()
    trainer.push_to_hub()

if __name__ == "__main__":
    finetune_llama_v2()