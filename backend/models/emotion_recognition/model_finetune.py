from datasets import load_from_disk, ClassLabel
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
import numpy as np
from evaluate import load
import torch

# Load dataset from disk or from Huggingface
dataset_path = "./data"
dataset = load_from_disk(dataset_path)

#from datasets import load_dataset
#ds = load_dataset("xbgoose/dusha")

print(dataset)

# Load the feature extractor from a pre-trained model
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

# Define a function to preprocess the audio data
def preprocess_function(examples):
    # Extract the raw audio arrays from the dataset
    audio_arrays = [x["array"] for x in examples["audio"]]
    
    # Use the feature extractor to process the audio
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=16000 * 5,  # Process up to 5 seconds of audio
        truncation=True,       # Truncate longer audio files
    )
    return inputs

# Apply the preprocessing function to the entire dataset
processed_dataset = dataset.map(preprocess_function, remove_columns="audio", batched=True)

# Get the unique labels from the train set
labels = sorted(list(dataset["train"].unique("emotion")))
print(f"The labels are: {labels}")

# Cast the "emotion" column to a ClassLabel type for the whole dataset
processed_dataset = processed_dataset.cast_column("emotion", ClassLabel(names=labels))

# Now that the column is a ClassLabel, we can create the mappings
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i  # Use integer IDs
    id2label[i] = label

# Load the pre-trained model and configure it for our task
model = AutoModelForAudioClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=len(labels),
    label2id=label2id,
    id2label=id2label,
)

processed_dataset = processed_dataset.rename_column("emotion", "labels")

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./ERS_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    fp16=True if torch.cuda.is_available() else False,
    fp16_full_eval=True,
    metric_for_best_model="eval_f1",
    save_total_limit=2,
    dataloader_num_workers=4,
)

# Define a function to compute metrics during evaluation
metric_acc = load("accuracy")
metric_f1 = load("f1")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = metric_acc.compute(predictions=preds, references=p.label_ids)["accuracy"]
    f1 = metric_f1.compute(predictions=preds, references=p.label_ids, average="macro")["f1"]
    return {"accuracy": acc, "f1": f1}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["test"],
    processing_class=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("./fine-tuned-emotion-model")
feature_extractor.save_pretrained("./fine-tuned-emotion-model")