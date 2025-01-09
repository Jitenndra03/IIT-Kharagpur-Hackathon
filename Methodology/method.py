import os
import pandas as pd
import fitz
import re
from datetime import datetime

def check_paper_format(pdf_path):
    """Check if research paper PDF meets formatting requirements."""
    required_sections = {
        'abstract': False,
        'introduction': False,
        'related work|background': False,
        'methodology|method|approach': False,
        'experiment|result|evaluation': False,
        'conclusion': False,
        'reference': False
    }
    
    try:
        doc = fitz.open(pdf_path)
        text_content = ""
        
        for page in doc:
            text_content += page.get_text()
            
        # Check section presence
        for section_pattern in required_sections:
            if re.search(section_pattern, text_content.lower()):
                required_sections[section_pattern] = True
                
        # Check numbered sections
        section_numbers = re.findall(r'\n\d+\s+[A-Z][a-zA-Z\s]+\n', text_content)
        numbered_sections = [s.strip() for s in section_numbers]
        
        missing_sections = [s for s, present in required_sections.items() if not present]
        has_numbered_sections = len(numbered_sections) >= 3
        
        return {
            "filename": os.path.basename(pdf_path),
            "status": "Pass" if not missing_sections and has_numbered_sections else "Fail",
            "missing_sections": ", ".join(missing_sections) if missing_sections else "None",
            "numbered_sections": ", ".join(numbered_sections),
            "has_numbered_sections": "Yes" if has_numbered_sections else "No"
        }
    
    except Exception as e:
        return {
            "filename": os.path.basename(pdf_path),
            "status": "Error",
            "missing_sections": f"Error: {str(e)}",
            "numbered_sections": "Error",
            "has_numbered_sections": "Error"
        }

def process_folder(folder_path):
    """Process all PDFs in a folder and generate report."""
    results = []
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        results.append(check_paper_format(pdf_path))
    
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"paper_format_check_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "Pass")
    failed = sum(1 for r in results if r["status"] == "Fail")
    errors = sum(1 for r in results if r["status"] == "Error")
    
    summary = f"""
Format Check Summary:
Total PDFs: {total}
Passed: {passed}
Failed: {failed}
Errors: {errors}
Results saved to: {csv_path}
    """
    print(summary)
    
    with open(f"format_summary_{timestamp}.txt", "w") as f:
        f.write(summary)

if __name__ == "__main__":
    folder_path = "/home/jitendra/Documents/IIT-Kharagpur-Hackathon/dataset"
    process_folder(folder_path)