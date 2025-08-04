# ------------------------------
# STEP 0: Install dependencies (run this once in terminal)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# pip install transformers datasets
# ------------------------------

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import json
import torch
from sklearn.model_selection import train_test_split
import torch

# ------------------------------
# STEP 1: Detect device (Apple M3 - MPS or CPU)
# ------------------------------
device = torch.device("mps") if torch.has_mps else torch.device("cpu")
print(f"✅ Using device: {device}")

# ------------------------------
# STEP 2: Generate synthetic dataset (500 examples now)
# ------------------------------
with open("./labelled_notes.json", "r") as file:
    data = json.load(file)


train_texts, val_texts, train_labels, val_labels = train_test_split(
    [d["text"] for d in data], [d["label"] for d in data], test_size=0.2
)
# ------------------------------
# STEP 3: Load model and tokenizer
# ------------------------------
MODEL_DIR = "./approach_openai_vision/ai_models/notes_model"

# tokenizer = AutoTokenizer.from_pretrained(
#     "bert-base-uncased"
# )  # trained on 110M parameters

# model = AutoModelForSequenceClassification.from_pretrained(
#     "bert-base-uncased", num_labels=2
# )

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=64)


# ------------------------------
# STEP 4: Tokenization
# ------------------------------
class NoteDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} | {
            "labels": torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)


train_dataset = NoteDataset(train_encodings, train_labels)
val_dataset = NoteDataset(val_encodings, val_labels)

# ------------------------------
# STEP 5: Training Arguments
# ------------------------------
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=6,
    warmup_steps=100,
    logging_dir="./logs",
    logging_steps=100,
    report_to="none",
)

# ------------------------------
# STEP 6: Trainer Setup
# ------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# ------------------------------
# STEP 7: Train the Model
# ------------------------------
trainer.train()

# ------------------------------
# STEP 8: Save Model
# ------------------------------
save_path = "./model_retrained"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"✅ Model saved successfully at {save_path}")


# ------------------------------
# STEP 9: Predict on New Prompt
# ------------------------------
def predict_note_header(text: str) -> dict:
    """
    Returns a dict with:
      - label: int (predicted class)
      - score: float (softmax probability)
      - logits: List[float] (raw model logits)
    """
    MODEL_DIR = "./model_retrained"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()  # set to inference mode

    # Tokenize + build batched tensors
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",  # pad to your chosen max_length
        max_length=64,
    )
    # If using GPU, uncomment:
    # inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]  # shape: [num_labels]
    probs = torch.nn.functional.softmax(logits, dim=-1)

    predicted_class = torch.argmax(probs).item()
    confidence = probs[predicted_class].item()

    # return {"label": predicted_class, "score": confidence, "logits": logits.tolist()}
    if predicted_class == 1:
        return True

    return False


# Example Inference

# if __name__ == '__main__':
#     test_prompt = "2.28 Function-wise classification of Statement of Profit and Loss"
#     print("\n🧠 Prediction:")
#     print(predict_note_header(test_prompt))
