import pandas as pd
import torch
import json # Import the json library

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# =========================================
# CHECK GPU
# =========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {device}")

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("WARNING: GPU not enabled")

# =========================================
# MODEL CONFIGURATION
# =========================================

MODEL_NAME = "distilbert-base-uncased"

# =========================================
# LOAD JSONL DATASET
# =========================================

print("\nLoading dataset from JSONL file...")

df_local = None # Initialize df_local to None
data_records = []

try:
    with open("all.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                data_records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"JSON decoding error on line {i+1}: {e}")
                print(f"Problematic line content: {line.strip()}")
                # Optionally, you can choose to skip problematic lines or exit
                # For now, we'll continue and see how many valid lines we get

    if data_records:
        df_local = pd.DataFrame(data_records)
        print("Dataset loaded successfully!")
    else:
        print("ERROR: No valid data records found in all.jsonl.")
        exit()

except FileNotFoundError:
    print("ERROR: all.jsonl file not found")
    exit()
except Exception as e:
    print(f"Dataset loading error: {e}")
    exit()

# =========================================
# PREPARE HUMAN + AI DATA
# =========================================

human_texts = []
ai_texts = []

print("\nPreparing dataset...")

# Ensure df_local is not None before proceeding
if df_local is None:
    print("ERROR: df_local is not defined due to previous loading issues. Exiting.")
    exit()

for index, row in df_local.iterrows():

    # HUMAN ANSWERS
    if (
        "human_answers" in row
        and row["human_answers"]
        and pd.notna(row["human_answers"][0])
    ):

        human_texts.append({
            "text": row["human_answers"][0],
            "label": 0
        })

    # AI ANSWERS
    if (
        "chatgpt_answers" in row
        and row["chatgpt_answers"]
        and pd.notna(row["chatgpt_answers"][0])
    ):

        ai_texts.append({
            "text": row["chatgpt_answers"][0],
            "label": 1
        })

# =========================================
# REDUCE DATASET SIZE
# =========================================

# Faster internship training

human_texts = human_texts[:1000]
ai_texts = ai_texts[:1000]

combined_data = human_texts + ai_texts

# =========================================
# CREATE DATAFRAME
# =========================================

df_processed = pd.DataFrame(combined_data)

print("\nProcessed Dataset:")
print(df_processed.head())

print("\nTotal Samples:", len(df_processed))

# =========================================
# CREATE HUGGING FACE DATASET
# =========================================

dataset = Dataset.from_pandas(df_processed)

print("\nDataset created successfully!")

print(dataset)

# =========================================
# LOAD TOKENIZER
# =========================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# =========================================
# TOKENIZATION FUNCTION
# =========================================

def tokenize_function(example):

    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=64
    )

# =========================================
# TOKENIZE DATASET
# =========================================

print("\nTokenizing dataset...")

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True
)

# REMOVE RAW TEXT
tokenized_dataset = tokenized_dataset.remove_columns(["text"])

# SET TORCH FORMAT
tokenized_dataset.set_format("torch")

# =========================================
# TRAIN / TEST SPLIT
# =========================================

tokenized_dataset = tokenized_dataset.train_test_split(
    test_size=0.2
)

train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]

print("\nTrain Dataset:")
print(train_dataset)

print("\nTest Dataset:")
print(test_dataset)

# =========================================
# LOAD MODEL
# =========================================

print("\nLoading DeBERTa model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.to(device)

# =========================================
# TRAINING CONFIGURATION
# =========================================

training_args = TrainingArguments(
    output_dir="./results",

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=1e-5,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    num_train_epochs=2,

    weight_decay=0.01,

    logging_steps=10,

    fp16=False,

    dataloader_num_workers=0,

    report_to="none"
)

# =========================================
# TRAINER SETUP
# =========================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# =========================================
# START TRAINING
# =========================================

print("\nStarting training...\n")

trainer.train()

# =========================================
# SAVE MODEL
# =========================================

print("\nSaving model...")

model.save_pretrained("./text_detector_model")

tokenizer.save_pretrained("./text_detector_model")

print("\nTraining completed successfully!")

print("\nModel saved inside:")
print("./text_detector_model")