<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
</head>
<body>
    <h1>Scholarly Paper Citation Fetcher</h1>
    <p>
        This Python script uses the <code>scholarly</code> library to fetch citation details for a specified research paper by its title. 
        It retrieves information about the paper's title, authors, publication year, venue, and the number of citations.
    </p>

  <h2>Features</h2>
    <ul>
        <li><strong>Search by Title:</strong> Finds the paper based on the given title using the <code>scholarly</code> library.</li>
        <li><strong>Paper Details:</strong> Displays details including authors, publication year, venue, and citation count.</li>
        <li><strong>Error Handling:</strong> Provides clear messages if the paper is not found or if an error occurs during the fetch process.</li>
    </ul>

  <h2>Requirements</h2>
    <p>
        Make sure the following Python package is installed:
    </p>
    <ul>
        <li><code>scholarly</code>: For querying Google Scholar data.</li>
    </ul>
    <p>Install it using pip:</p>
    <pre><code>pip install scholarly</code></pre>

   <h2>Usage</h2>
    <ol>
        <li><strong>Run the Script:</strong>
            <pre><code>python script_name.py</code></pre>
        </li>
        <li><strong>Enter the Paper Title:</strong>
            <p>When prompted, input the title of the paper you want to fetch citation details for.</p>
        </li>
        <li><strong>View Results:</strong>
            <p>The script will display the following details if the paper is found:</p>
            <ul>
                <li>Title</li>
                <li>Authors</li>
                <li>Publication Year</li>
                <li>Venue</li>
                <li>Number of Citations</li>
            </ul>
        </li>
    </ol>

  <h2>Example Output</h2>
    <pre><code>
Enter the paper title: A Study on Neural Networks
Title: A Study on Neural Networks
Authors: John Doe, Jane Smith
Published Year: 2021
Venue: Journal of Artificial Intelligence
Cited By: 123
    </code></pre>

  <h2>Error Handling</h2>
    <ul>
        <li>If the paper title is not found, the script will display: <code>Paper not found. Please check the title and try again.</code></li>
        <li>If an unexpected error occurs, an appropriate error message will be displayed.</li>
    </ul>

  <h2>License</h2>
    <p>This project is open source and available under the MIT License.</p>
</body>
</html>

