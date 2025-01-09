<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <h1>Plagiarism Checker and Publishability Assessment</h1>
    <p>
        This Python script checks a given PDF file for plagiarism against a set of source text files and evaluates its readability for publishability. 
        It generates a comprehensive report that includes plagiarism comparisons with source papers and an assessment of the document's readability.
    </p>

  <h2>Features</h2>
    <ul>
        <li><strong>Plagiarism Check:</strong> Compares the input PDF with source text files to calculate the similarity percentage.</li>
        <li><strong>Readability Assessment:</strong> Calculates the Flesch Reading Ease score and estimates reading time to assess if the document is publishable.</li>
        <li><strong>Comprehensive Report:</strong> Generates a final report containing both plagiarism results and publishability assessment, and saves it as a text file.</li>
    </ul>

   <h2>Requirements</h2>
    <p>
        Before running the script, make sure you have the following Python packages installed:
    </p>
    <ul>
        <li><code>PyPDF2</code> for PDF text extraction</li>
        <li><code>textstat</code> for readability analysis</li>
    </ul>
    <p>You can install the required dependencies using <code>pip</code>:</p>
    <pre><code>pip install PyPDF2 textstat</code></pre>

   <h2>File Structure</h2>
    <ul>
        <li><strong>input_paper.pdf:</strong> The PDF file you want to check for plagiarism and readability.</li>
        <li><strong>source_papers/:</strong> Folder containing source <code>.txt</code> files to compare the input PDF against.</li>
        <li><strong>final_report.txt:</strong> The file where the final plagiarism and publishability report will be saved.</li>
        <li><strong>main.py:</strong> The Python script that performs the plagiarism check and readability assessment.</li>
    </ul>

   <h2>Usage</h2>
    <ol>
        <li><strong>Prepare Input Files:</strong>
            <ul>
                <li>Place the PDF file you want to analyze in the same directory as the script (or adjust the <code>input_file</code> variable in the script).</li>
                <li>Store the source <code>.txt</code> files in the <code>source_papers/</code> directory (or modify the <code>source_folder</code> path in the script).</li>
            </ul>
        </li>
        <li><strong>Run the Script:</strong>
            <pre><code>python main.py</code></pre>
        </li>
        <li><strong>Review the Report:</strong>
            <p>The script will generate and save a <code>final_report.txt</code> file, which includes:</p>
            <ul>
                <li><strong>Plagiarism Report:</strong> A comparison of the input text with each source text and the similarity percentage.</li>
                <li><strong>Publishability Assessment:</strong> A summary of the Flesch Reading Ease score and reading time, and whether the document is considered publishable.</li>
            </ul>
            <p>The report will also be printed to the console.</p>
        </li>
    </ol>

   <h2>Example Output</h2>
    <pre><code>
Extracting text from PDF...
Checking plagiarism...
Assessing publishability...

--- Final Report ---
Plagiarism Report:
- Compared with paper1.txt: 75.23% similarity
- Compared with paper2.txt: 12.45% similarity
- Compared with paper3.txt: 58.00% similarity

Publishability Assessment:
- Flesch Reading Ease: 60.32
- Estimated Reading Time: 9.47 minutes
- Publishable: Yes

Report saved as 'final_report.txt'.
    </code></pre>

   <h2>Error Handling</h2>
    <ul>
        <li>The script checks if the input PDF file and the source folder exist. If either is missing, an error message will be displayed.</li>
        <li>If there is an issue reading the PDF file, an error message will be printed, and no report will be generated.</li>
    </ul>

   <h2>License</h2>
    <p>This project is open source and available under the MIT License.</p>
</body>
</html>
