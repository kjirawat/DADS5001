# Text Summarization and Analysis Web App

This web application utilizes the Hugging Face API for text summarization, along with various analysis tools such as word count, word frequency visualization, and word clouds. Users can input text, and the app provides a summary of the input text along with various insights into its content.

## Features

- **Text Input**: Users can input text into the provided text area.
- **Text Summarization**: Utilizes the Hugging Face API (Facebook's BART large model trained on CNN data) to summarize the input text.
- **Word Count**: Displays the word count of both the input text and its summary.
- **Word Frequency Scatter Plot**: Visualizes the frequency of words in both the input text and its summary.
- **Word Clouds**: Generates word clouds for the input text and its summary, providing a visual representation of word frequency.


  Example from using text input from https://edition.cnn.com/2024/04/19/business/cinemark-drink-lawsuit/index.html

  
![Visualization of the dataset](https://github.com/ratimayy/tools/blob/main/pic1.png)
![Visualization of the dataset](https://github.com/ratimayy/tools/blob/main/pic2.png)
![Visualization of the dataset](https://github.com/ratimayy/tools/blob/main/pic3.png)
![Visualization of the dataset](https://github.com/ratimayy/tools/blob/main/pic4.png)
## Usage

1. Enter text in the provided text area.
2. Click the "Submit" button to generate the summary and analysis.
3. The output will include the summarized text, word count comparison, word frequency scatter plots, and word clouds.

## Dependencies

- Dash: A Python framework for building analytical web applications.
- Plotly: A Python graphing library that makes interactive, publication-quality graphs.
- Requests: A simple HTTP library for Python.
- WordCloud: A Python library for creating word clouds.


## Contributors
6610422001  Ratima  Sattaolankit


6610422021  Nattachai Jairak


6610422024  Jirawat Kemsatitanan
