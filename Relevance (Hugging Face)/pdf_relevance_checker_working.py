import os
import PyPDF2
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline
)

def extract_text_from_pdfs(pdf_directory):
    texts = []
    for file_name in os.listdir(pdf_directory):
        if file_name.endswith('.pdf'):
            with open(os.path.join(pdf_directory, file_name), 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() + ' '
            texts.append(text)
    return texts

def prepare_dataset(pdf_directory):
    pdf_texts = extract_text_from_pdfs(pdf_directory)
    # Example: Create labels (1 for publishable, 0 for non-publishable)
    labels = [1 if 'relevant' in text.lower() else 0 for text in pdf_texts]
    return pdf_texts, labels

class PDFDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def truncate_text(text, tokenizer, max_length=512):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
    return tokenizer.decode(tokens, skip_special_tokens=True)

def main(pdf_directory):
    # Prepare dataset
    pdf_texts, labels = prepare_dataset(pdf_directory)
    
    # Load tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    # Truncate texts before splitting
    pdf_texts = [truncate_text(text, tokenizer) for text in pdf_texts]

    # Split dataset
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        pdf_texts, labels, test_size=0.2, random_state=42
    )

    # Tokenize datasets
    train_encodings = tokenizer(
        train_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).data
    
    val_encodings = tokenizer(
        val_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).data

    # Create datasets
    train_dataset = PDFDataset(train_encodings, train_labels)
    val_dataset = PDFDataset(val_encodings, val_labels)

    # Initialize model
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        learning_rate=5e-5,
        weight_decay=0.01,
    )

    # Train model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    # Save model
    model.save_pretrained('./model')
    tokenizer.save_pretrained('./model')

    # Inference
    classifier = pipeline('text-classification', model='./model', tokenizer=tokenizer)

    for text in pdf_texts:
        # Ensure text is truncated
        truncated_text = truncate_text(text, tokenizer)
        result = classifier(truncated_text)
        print(result)
 
if __name__ == "__main__":
    pdf_directory = "D:\Tools For Hackathon\Relevance (Hugging Face)\ResearchPapers"
    main(pdf_directory)