import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, create_optimizer
from sklearn.model_selection import train_test_split
from datasets import load_dataset

# Load the tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

# Load and preprocess dataset
def preprocess_data(texts, labels, tokenizer, max_length):
    encodings = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="tf"
    )
    dataset = tf.data.Dataset.from_tensor_slices((
        {"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"]},
        labels
    ))
    return dataset

# Example: Loading a dataset (you can replace this with your custom dataset)
data = load_dataset("imdb", split="train")
texts = data["text"]
labels = data["label"]

# Train-validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Preprocess train and validation data
max_length = 128
train_dataset = preprocess_data(train_texts, train_labels, tokenizer, max_length)
val_dataset = preprocess_data(val_texts, val_labels, tokenizer, max_length)

# Set batch size and prepare datasets for training
batch_size = 16
train_dataset = train_dataset.shuffle(len(train_texts)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Load the BERT model
num_labels = 2
model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Compile the model
num_train_steps = len(train_dataset) * 3  # 3 epochs
optimizer, schedule = create_optimizer(
    init_lr=5e-5, num_warmup_steps=0, num_train_steps=num_train_steps
)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ["accuracy"]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Train the model
epochs = 3
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

# Save the model and tokenizer
model.save_pretrained("./Model")
tokenizer.save_pretrained("./Tokenizer")

print("Model and tokenizer saved!")
