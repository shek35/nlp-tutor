import torch
import datasets
import pandas as pd
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)


base_model = "mistralai/Mistral-7B-Instruct-v0.2"

# Load the data
data_files = [
    "/textbooks_text/1709.07809-2.txt",
    "/textbooks_text/ed3bookfeb3_2024-2.txt",
    "/textbooks_text/eisenstein-nlp-notes-2.txt",
    "/textbooks_text/mml-book-2.txt",
]

data = []

for data_file in data_files:
    with open(data_file, "r") as f:
        data += f.readlines()

# Remove the hyphenation
s = ""
for i in data:
    if i.endswith("-\n"):
        s += i.replace("-\n", "").strip()
    else:
        s += i + " "

# Remove the newlines
s = s.replace("\n", "")

# Split the text into chunks of 2048 tokens
s_list = s.split()

chunks = []
for i in range(0, len(s_list), 2048):
    chunks.append(" ".join(s_list[i : i + 2048]))

df = pd.DataFrame(chunks, columns=["text"])

# Create the dataset
dataset = datasets.Dataset.from_pandas(df)

# Free the memory
del data, s, s_list, chunks, df


# Load the Mistral 7B model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model, quantization_config=bnb_config, device_map={"": 0}
)

model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token


# Prepare the model for training
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
)
model = get_peft_model(model, peft_config)


training_arguments = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=2,
    optim="paged_adamw_8bit",
    save_steps=50,
    logging_steps=20,
    learning_rate=1e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.3,
    group_by_length=True,
    lr_scheduler_type="constant",
)

# Create the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=None,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

# Train the model
trainer.train()

# Save the model
trainer.model.save_pretrained("./model")
