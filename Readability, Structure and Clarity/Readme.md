<h1>Readability, Structure, and Clarity Checker</h1>

<p>This program analyzes a research paper for readability, structure, and clarity. It evaluates the content based on the <strong>grammar error percentage</strong> and other metrics.</p>

<h2>Installation</h2>
<p>Before running the program, ensure the following Python libraries are installed:</p>
<pre><code>pip install PyPDF2
pip install syllapy
pip install spacy
pip install language_tool_python
</code></pre>

<h2>Usage</h2>
<p>To check a research paper, provide the path to the PDF file as input. For example:</p>
<pre><code>Enter the path to the research paper (PDF): non1.pdf</code></pre>

<h2>Sample Output</h2>
<p>Given an input file <code>non1.pdf</code>, the program generates the following output:</p>
<pre><code>
Flesch-Kincaid Grade Level: 28.27
The text is very difficult to read.
Is the document well-structured? True
It is not publishable
Grammar Error Percentage: 0.91%
</code></pre>

<h2>Key Features</h2>
<ul>
  <li><strong>Readability:</strong> Calculates the Flesch-Kincaid Grade Level.</li>
  <li><strong>Structure:</strong> Checks if the document is well-structured.</li>
  <li><strong>Clarity:</strong> Measures the grammar error percentage.</li>
</ul>

<h2>Conclusion</h2>
<p>The program helps evaluate whether a research paper is <em>publishable</em> based on the readability, structure, and clarity metrics.</p>
