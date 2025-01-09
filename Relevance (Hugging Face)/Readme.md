<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Relevance Classifier</title>
</head>
<body>
    <h1>PDF Relevance Classifier</h1>
    <p>
        This project uses a fine-tuned RoBERTa model to classify research papers (PDFs) as "publishable" or "non-publishable" based on their content. The pipeline extracts text from PDFs, prepares datasets, trains a model, and performs inference.
    </p>

  <h2>Features</h2>
    <ul>
        <li><strong>PDF Text Extraction:</strong> Extracts text from all PDF files in a specified directory.</li>
        <li><strong>Dataset Preparation:</strong> Labels data based on content relevance.</li>
        <li><strong>Model Training:</strong> Fine-tunes a <code>roberta-base</code> model on the prepared dataset.</li>
        <li><strong>Inference Pipeline:</strong> Predicts the relevance of new documents.</li>
    </ul>

  <h2>Installation</h2>
    <ol>
        <li>Clone the repository:
            <pre><code>git clone &lt;repository_url&gt;
cd &lt;repository_folder&gt;</code></pre>
        </li>
        <li>Install the required Python libraries:
            <pre><code>pip install torch transformers PyPDF2 scikit-learn</code></pre>
        </li>
        <li>Ensure you have a compatible GPU (optional but recommended for faster training).</li>
    </ol>

  <h2>Usage</h2>
    <ol>
        <li>Prepare a directory containing research papers in PDF format.</li>
        <li>Update the <code>pdf_directory</code> variable in the script:
            <pre><code>pdf_directory = "path/to/your/pdf/directory"</code></pre>
        </li>
        <li>Run the script:
            <pre><code>python script_name.py</code></pre>
        </li>
        <li>Example output during inference:
            <pre><code>[{'label': 'LABEL_1', 'score': 0.85}]
[{'label': 'LABEL_0', 'score': 0.92}]</code></pre>
        </li>
    </ol>

  <h2>Workflow</h2>
    <ol>
        <li><strong>Text Extraction:</strong> The <code>extract_text_from_pdfs</code> function reads all PDF files in the specified directory and extracts their text content.</li>
        <li><strong>Dataset Preparation:</strong> Assigns binary labels (<code>1</code> for "publishable" and <code>0</code> for "non-publishable") based on the presence of specific keywords.</li>
        <li><strong>Text Truncation:</strong> Ensures texts fit within the 512-token limit of the RoBERTa model.</li>
        <li><strong>Model Training:</strong> Fine-tunes the <code>roberta-base</code> model using the Hugging Face <code>Trainer</code> API.</li>
        <li><strong>Inference:</strong> Classifies texts and outputs predictions.</li>
    </ol>

  <h2>Directory Structure</h2>
    <pre><code>.
├── script_name.py      # Main script for training and inference
├── results/            # Model training results
├── logs/               # Logs generated during training
├── model/              # Saved fine-tuned model and tokenizer
├── ResearchPapers/     # Directory containing input PDFs
</code></pre>

  <h2>Training Parameters</h2>
    <ul>
        <li>Model: <code>roberta-base</code></li>
        <li>Number of epochs: <code>3</code></li>
        <li>Batch size: <code>8</code></li>
        <li>Learning rate: <code>5e-5</code></li>
        <li>Weight decay: <code>0.01</code></li>
    </ul>

  <h2>Output</h2>
    <p>
        After running the script:
        <ul>
            <li>The fine-tuned model is saved in the <code>model/</code> directory.</li>
            <li>The results of inference are printed to the console.</li>
        </ul>
    </p>

  <h2>Example Inference</h2>
    <p>
        For a PDF containing the following text:
        <pre><code>"This research paper contains relevant data on artificial intelligence."</code></pre>
        The output might be:
        <pre><code>[{'label': 'LABEL_1', 'score': 0.98}]</code></pre>
    </p>

  <h2>Dependencies</h2>
    <ul>
        <li><code>torch</code></li>
        <li><code>transformers</code></li>
        <li><code>PyPDF2</code></li>
        <li><code>scikit-learn</code></li>
    </ul>

  <h2>License</h2>
    <p>
        This project is licensed under the MIT License. See the LICENSE file for details.
    </p>
</body>
</html>
