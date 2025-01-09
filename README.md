<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hackathon IITKh Round 2 - Readme</title>
</head>
<body>
    <h1>Hackathon IITKh Round 2</h1>
    <p>
        This project evaluates the quality of PDF research papers based on multiple factors such as readability, structure, clarity, relevance, and impact. It utilizes various tools and libraries to perform these checks.
    </p>

  <h2>Key Features</h2>
    <ul>
        <li><strong>Readability:</strong> Assessed using the Syllapy library.</li>
        <li><strong>Clarity:</strong> Evaluated with the Spacy library.</li>
        <li><strong>Structure:</strong> Checked for proper organization and coherence using PyPDF2.</li>
        <li><strong>Relevance:</strong> Determined based on the content's alignment with the topic.</li>
        <li><strong>Impact:</strong> Statistical analysis and methodology checks.</li>
    </ul>

  <h2>Dependencies</h2>
    <ul>
        <li><code>PyPDF2</code>: Extracts text and checks structure.</li>
        <li><code>Syllapy</code>: Measures readability.</li>
        <li><code>Spacy</code>: Checks clarity and coherence.</li>
        <li><code>torch</code>: Supports machine learning models for relevance and impact assessment.</li>
    </ul>

  <h2>Evaluation Output</h2>
    <p>The program provides results for the following metrics:</p>
    <ul>
        <li><strong>Content Quality:</strong> Clarity, structure, readability, and impact.</li>
        <li><strong>Methodology Check:</strong> Validates statistical analyses and citations.</li>
        <li><strong>Relevance:</strong> Determines if the content is aligned with the intended topic.</li>
    </ul>
    <p>Each metric is scored individually, and a decision is made whether the paper is <strong>Publishable</strong> or <strong>Non-Publishable</strong>.</p>

  <h2>Sample Output</h2>
    <pre><code>
Readability: Average
Clarity: Average
Structure: Average
Relevance: Inappropriate methodologies
Impact: Incoherent arguments
Final Decision: Non-Publishable
    </code></pre>

  <h2>How to Use</h2>
    <ol>
        <li>Prepare a directory containing research papers in PDF format.</li>
        <li>Run the script and provide the directory path as input.</li>
        <li>Review the results for each PDF, including metrics and the final decision.</li>
    </ol>

  <h2>Conclusion</h2>
    <p>
        This tool is designed to assist researchers and editors in determining the quality of research papers. It automates the evaluation process, saving time and improving consistency in assessments.
    </p>

  <h2>License</h2>
    <p>
        This project is licensed under the MIT License. See the LICENSE file for more details.
    </p>
</body>
</html>
