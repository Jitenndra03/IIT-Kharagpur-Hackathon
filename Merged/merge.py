import os
import re
import fitz
import torch
import difflib
import textstat
import pandas as pd
from datetime import datetime
from PyPDF2 import PdfReader
from transformers import (RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments, pipeline)
import language_tool_python
import spacy

nlp = spacy.load("en_core_web_sm")


# Section 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""


# Section 2: Check formatting and structure
def check_paper_format(pdf_text):
    required_sections = {
        'abstract': False,
        'introduction': False,
        'related work|background': False,
        'methodology|method|approach': False,
        'experiment|result|evaluation': False,
        'conclusion': False,
        'reference': False
    }

    for section_pattern in required_sections:
        if re.search(section_pattern, pdf_text.lower()):
            required_sections[section_pattern] = True

    missing_sections = [s for s, present in required_sections.items() if not present]
    return {
        "status": "Pass" if not missing_sections else "Fail",
        "missing_sections": ", ".join(missing_sections) if missing_sections else "None"
    }


# Section 3: Citation extraction and validation
def extract_citations(text):
    # Common citation patterns (IEEE, APA, etc.)
    patterns = [
        r'\[[0-9]+\]',  # IEEE style [1], [2], etc.
        r'\([A-Za-z]+, \d{4}\)',  # APA style (Author, 2020)
        r'[A-Z][a-z]+ et al., \d{4}',  # MLA style (Author et al., 2020)
    ]
    citations = []
    for pattern in patterns:
        citations.extend(re.findall(pattern, text))
    return citations


def validate_citations(citations, threshold=5):
    # Minimum threshold of citations for publishability
    return {
        "citation_count": len(citations),
        "status": "Pass" if len(citations) >= threshold else "Fail",
        "threshold": threshold
    }


# Section 4: Readability and grammar analysis
def readability_analysis(text):
    readability_score = textstat.flesch_reading_ease(text)
    reading_time = textstat.reading_time(text)
    return readability_score, reading_time


def grammar_error_analysis(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    total_words = len(text.split())
    error_percentage = (len(matches) / total_words) * 100
    return error_percentage


# Section 5: Plagiarism check
def check_plagiarism(input_text, source_texts):
    report = []
    for source_name, content in source_texts.items():
        similarity = difflib.SequenceMatcher(None, input_text, content).ratio() * 100
        report.append(f"Compared with {source_name}: {similarity:.2f}% similarity")
    return "\n".join(report)


# Section 6: Machine learning model for publishability
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


def load_ml_model():
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')  # Use model from Hugging Face
    return tokenizer, model



def predict_publishability(text, tokenizer, model):
    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)
    result = classifier(text[:512])  # Truncate to max token length
    return result


# Section 7: Main pipeline
def assess_publishability(pdf_path, source_folder):
    # Extract text
    print("Extracting text from PDF...")
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        return "Error extracting text."

    # Check formatting and structure
    print("Checking format and structure...")
    format_result = check_paper_format(pdf_text)

    # Extract and validate citations
    print("Extracting and validating citations...")
    citations = extract_citations(pdf_text)
    citation_result = validate_citations(citations)

    # Readability analysis
    print("Analyzing readability...")
    readability_score, reading_time = readability_analysis(pdf_text)

    # Grammar analysis
    print("Analyzing grammar...")
    grammar_error_percentage = grammar_error_analysis(pdf_text)

    # Load source texts for plagiarism check
    print("Checking plagiarism...")
    source_texts = {}
    for filename in os.listdir(source_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(source_folder, filename), 'r', encoding="utf-8") as file:
                source_texts[filename] = file.read()
    plagiarism_report = check_plagiarism(pdf_text, source_texts)

    # ML-based publishability assessment
    print("Predicting publishability with ML...")
    tokenizer, model = load_ml_model()
    ml_prediction = predict_publishability(pdf_text, tokenizer, model)

    # Generate final report
    print("Generating final report...")
    report = f"""
    --- Publishability Report ---
    Format Check: {format_result['status']}
    Missing Sections: {format_result['missing_sections']}
    Citation Check: {citation_result['status']}
    Citation Count: {citation_result['citation_count']} (Threshold: {citation_result['threshold']})
    Readability Score (Flesch): {readability_score:.2f}
    Estimated Reading Time: {reading_time:.2f} minutes
    Grammar Error Percentage: {grammar_error_percentage:.2f}%
    ML Prediction: {ml_prediction}

    --- Plagiarism Report ---
    {plagiarism_report}
    """
    return report


if __name__ == "__main__":
    pdf_path = "/home/jitendra/Documents/IIT-Kharagpur-Hackathon/dataset/Reference/Publishable/NeurIPS/R013.pdf"
    source_folder = "/home/jitendra/Documents/IIT-Kharagpur-Hackathon/dataset/Papers"
    result = assess_publishability(pdf_path, source_folder)
    print(result)
    with open("publishability_report.txt", "w") as file:
        file.write(result)
