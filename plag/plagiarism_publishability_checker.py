import os
import difflib
import textstat
from PyPDF2 import PdfReader

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return ""

# Function to load source papers
def load_source_papers(folder_path):
    """Loads all source papers from a folder."""
    source_texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding="utf-8") as file:
                source_texts[filename] = file.read()
    return source_texts

# Function to calculate similarity
def calculate_similarity(text1, text2):
    """Calculates similarity percentage between two texts."""
    text1_lines = text1.splitlines()
    text2_lines = text2.splitlines()
    similarity = difflib.SequenceMatcher(None, text1_lines, text2_lines).ratio()
    return similarity * 100  # Convert to percentage

# Function to check plagiarism
def check_plagiarism(input_text, source_texts):
    """Checks plagiarism of an input text against source papers."""
    report = "Plagiarism Report:\n"
    for source, content in source_texts.items():
        similarity = calculate_similarity(input_text, content)
        report += f"- Compared with {source}: {similarity:.2f}% similarity\n"
    return report

# Function to assess readability and publishability
def assess_publishability(input_text):
    """Assesses readability and decides publishability."""
    flesch_score = textstat.flesch_reading_ease(input_text)
    reading_time = textstat.reading_time(input_text)

    publishable = flesch_score > 50 and reading_time < 15  # Example criteria
    report = (
        "Publishability Assessment:\n"
        f"- Flesch Reading Ease: {flesch_score:.2f}\n"
        f"- Estimated Reading Time: {reading_time:.2f} minutes\n"
        f"- Publishable: {'Yes' if publishable else 'No'}\n"
    )
    return report

# Main function
def main():
    # File paths
    source_folder = "source_papers/"  # Folder containing source papers
    input_file = "input_paper.pdf"    # PDF file to check

    # Extract text from input PDF
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' not found!")
        return

    print("Extracting text from PDF...")
    input_text = extract_text_from_pdf(input_file)
    if not input_text:
        print("Failed to extract text from the input PDF!")
        return

    # Load source papers
    if not os.path.exists(source_folder):
        print(f"Source folder '{source_folder}' not found!")
        return

    source_texts = load_source_papers(source_folder)

    # Run plagiarism check
    print("Checking plagiarism...")
    plagiarism_report = check_plagiarism(input_text, source_texts)

    # Run publishability assessment
    print("Assessing publishability...")
    publishability_report = assess_publishability(input_text)

    # Combine reports
    final_report = plagiarism_report + "\n" + publishability_report

    # Save report to file
    with open("final_report.txt", 'w', encoding="utf-8") as file:
        file.write(final_report)

    print("\n--- Final Report ---")
    print(final_report)
    print("Report saved as 'final_report.txt'.")

if __name__ == "__main__":
    main()
