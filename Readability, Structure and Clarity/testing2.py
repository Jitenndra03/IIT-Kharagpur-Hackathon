import PyPDF2
import re
import syllapy
import spacy
import language_tool_python

nlp = spacy.load("en_core_web_sm")

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Function to calculate Flesch-Kincaid readability score
def flesch_kincaid_grade_level(text):
    # Split text into sentences and words
    sentences = re.split(r'[.!?]+', text)
    words = re.findall(r'\w+', text)
    
    # Calculate number of syllables
    syllables = sum(syllapy.count(word) for word in words)
    
    # Number of sentences and words
    num_sentences = len(sentences)
    num_words = len(words)

    # Avoid division by zero
    if num_sentences == 0 or num_words == 0:
        return None

    # Calculate Flesch-Kincaid Grade Level
    grade_level = (0.39 * (num_words / num_sentences)) + (11.8 * (syllables / num_words)) - 15.59
    return grade_level

def analyze_structure(text):
    doc=nlp(text)
    section_keywords = ["introduction", "methodology", "results", "discussion", "conclusion"]
    sections = []
    for sent in doc.sents:
        if any(keyword in sent.text.lower() for keyword in section_keywords):
            sections.append(sent.text)
    return sections
def is_well_structured(sections):
    """
    Determines if the document is well-structured based on detected sections.
    
    Args:
        sections (list): List of detected section headers.
    
    Returns:
        bool: True if well-structured, False otherwise.
    """
    if len(sections) >= 4:  # Adjust threshold as needed
        return True
    return False

def calculate_grammar_error_percentage(text, total_words):
    # Load the LanguageTool model for English
    tool = language_tool_python.LanguageTool('en-US')
    
    # Check for grammar errors
    matches = tool.check(text)
    
    # Count the number of errors
    error_count = len(matches)
    
    # Calculate the percentage of errors
    error_percentage = (error_count / total_words) * 100
    
    return error_percentage



# Main function
def main(pdf_path):
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)

    # Calculate readability score
    readability_score = flesch_kincaid_grade_level(text)
    analyze_structure(text)
    # Output the readability score
    if readability_score is not None:
        print(f"Flesch-Kincaid Grade Level: {readability_score:.2f}")
        if readability_score < 5:
            print("The text is very easy to read.")
        elif readability_score < 9:
            print("The text is easy to read.")
        elif readability_score < 13:
            print("The text is fairly difficult to read.")
        else:
            print("The text is very difficult to read.")
    else:
        print("Unable to calculate readability score.")
    sections = analyze_structure(text)
    print("Is the document well-structured?", is_well_structured(sections))  
    if(readability_score<9 and is_well_structured(sections)):
     print("It is publishable ")
    else:
        print("It is not publishable")
   
    total_words = len(text.split())
# Run the grammar check
    error_percentage = calculate_grammar_error_percentage(text, total_words)

# Output the percentage of grammar errors
    print(f"Grammar Error Percentage: {error_percentage:.2f}%")


    

# Example usage
if __name__ == "__main__":
    pdf_path = input("Enter the path to the research paper (PDF): ")
    main(pdf_path)  


   










    

