from datetime import datetime;
print(datetime.now(), "step 0 import library")

import dash
from dash import dcc, html, Input, Output
import requests
import plotly.graph_objs as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
import numpy as np
from io import BytesIO

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print(datetime.now(), "step 0 finish")

print(datetime.now(), "step 1 load pipeline")

# Load translation model and tokenizer
translator = pipeline("translation_xx_to_yy", model="facebook/mbart-large-50-many-to-many-mmt", tokenizer="facebook/mbart-large-50-many-to-many-mmt")

# Load summarization pipelines
summarizer_hf = pipeline("summarization", model="facebook/bart-large-cnn")
summarizer_custom = pipeline("summarization", model="Falconsai/text_summarization")

# Load sentence transformer model
model_st = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Define the sentiment analysis pipeline
# sentiment_pipe = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

print(datetime.now(), "step 1 finish")

print(datetime.now(), "step 2 create Dash app object and define functions")

# API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
API_URL = "https://api-inference.huggingface.co/models/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

# Change Authorization from "Bearer hf_xxx" to your access token, more information see:
# https://huggingface.co/docs/hub/en/security-tokens#what-are-user-access-tokens
headers = {"Authorization": "Bearer hf_xxx"}

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Textarea(
        id='input-text',
        placeholder='Enter your text here...',
        style={'width': '100%', 'height': 200}
    ),
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='sentiment-results'),
    html.Div(id='output-container-hf'),
    html.Div(id='output-container-custom'),
    html.Div(id='output-container-thai-hf'),  # Thai translation output from Facebook/Bart
    html.Div(id='output-container-thai-custom'),  # Thai translation output from Falconsai
    html.Div(id='sentence-similarity-hf'),  # Sentence similarity from Facebook/Bart
    html.Div(id='sentence-similarity-custom'),  # Sentence similarity from Falconsai
    dcc.Graph(id='word-count-graph'),
    html.Div([
        html.Div([
            html.H3("Word Cloud for Input Text"),
            html.Div(id='wordcloud-container-input'),
        ], style={'width': '33%', 'display': 'inline-block'}),
        html.Div([
            html.H3("Word Cloud for Summary HF"),
            html.Div(id='wordcloud-container-hf'),
        ], style={'width': '33%', 'display': 'inline-block'}),
        html.Div([
            html.H3("Word Cloud for Summary Custom"),
            html.Div(id='wordcloud-container-custom'),
        ], style={'width': '33%', 'display': 'inline-block'}),
    ])
])

def query(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def count_words(text):
    return len(text.split())

def generate_word_cloud(text):
    wordcloud = WordCloud(width=400, height=400).generate(text)
    return wordcloud.to_image()

def encode_image(image):
    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    img_str = "data:image/png;base64," + base64.b64encode(img_buffer.getvalue()).decode()
    return img_str

print(datetime.now(), "step 2 finish")

print(datetime.now(), "step 3 define callback")

@app.callback(
    [Output('sentiment-results', 'children'),
     Output('output-container-hf', 'children'),
     Output('output-container-custom', 'children'),
     Output('output-container-thai-hf', 'children'),
     Output('output-container-thai-custom', 'children'),
     Output('sentence-similarity-hf', 'children'),
     Output('sentence-similarity-custom', 'children'),
     Output('word-count-graph', 'figure'),
     Output('wordcloud-container-input', 'children'),
     Output('wordcloud-container-hf', 'children'),
     Output('wordcloud-container-custom', 'children')],
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('input-text', 'value')]
)
def update_output(n_clicks, input_text):
    print(datetime.now(), "update_output point 1, n_clicks=", n_clicks)
    
    if n_clicks > 0 and input_text:
        # Perform sentiment analysis
        # sentiment_output = sentiment_pipe(input_text)
        sentiment_temp = query({"inputs": input_text})
        sentiment_output = sentiment_temp[0]
        # print(sentiment_output)
        sentiment_result0 = sentiment_output[0]['score']
        sentiment_label0 = sentiment_output[0]['label']
        sentiment_result1 = sentiment_output[1]['score']
        sentiment_label1 = sentiment_output[1]['label']
        sentiment_result2 = sentiment_output[2]['score']
        sentiment_label2 = sentiment_output[2]['label']
        str1 = f"Sentiment: {sentiment_label0} ({sentiment_result0:.2f})"
        str2 = f"Sentiment: {sentiment_label1} ({sentiment_result1:.2f})"
        str3 = f"Sentiment: {sentiment_label2} ({sentiment_result2:.2f})"
        sentiment_result_formatted = str1 + ' ' + str2 + ' ' + str3

        # Summarization outputs
        output_hf = summarizer_hf(input_text, max_length=150, min_length=30, do_sample=False)
        output_custom = summarizer_custom(input_text, max_length=150, min_length=30, do_sample=False)
        summary_hf = output_hf[0]['summary_text']
        summary_custom = output_custom[0]['summary_text']
        input_word_count = count_words(input_text)
        summary_word_count_hf = count_words(summary_hf)
        summary_word_count_custom = count_words(summary_custom)

        # Translation to Thai
        translated_text_hf = translator(summary_hf, src_lang="en_XX", tgt_lang="th_TH")[0]['translation_text']
        translated_text_custom = translator(summary_custom, src_lang="en_XX", tgt_lang="th_TH")[0]['translation_text']

        # Calculate sentence similarity
        embeddings_input = model_st.encode([input_text])
        embeddings_summary_hf = model_st.encode([summary_hf])
        embeddings_summary_custom = model_st.encode([summary_custom])

        similarity_hf = cosine_similarity(embeddings_input, embeddings_summary_hf)[0][0]
        similarity_custom = cosine_similarity(embeddings_input, embeddings_summary_custom)[0][0]

        # Graph data
        data = [
            go.Bar(x=['Input Text', 'Summary HF', 'Summary Custom'], y=[input_word_count, summary_word_count_hf,
                                                                        summary_word_count_custom],
                   marker_color=['blue', 'orange', 'green'])
        ]
        layout = go.Layout(title="Word Counts", barmode='group')

        # Generate word clouds
        input_wordcloud_img = generate_word_cloud(input_text)
        hf_wordcloud_img = generate_word_cloud(summary_hf)
        custom_wordcloud_img = generate_word_cloud(summary_custom)

        input_wordcloud_data = html.Img(src=encode_image(input_wordcloud_img), style={'width': '75%'})
        hf_wordcloud_data = html.Img(src=encode_image(hf_wordcloud_img), style={'width': '75%'})
        custom_wordcloud_data = html.Img(src=encode_image(custom_wordcloud_img), style={'width': '75%'})

        print(datetime.now(), "update_output point 2, n_clicks=", n_clicks)
        return html.Div([
            html.H3("Sentiment Analysis Result:"),
            html.P(sentiment_result_formatted)
        ]), html.Div([
            html.H3("Output from Facebook/Bart:"),
            html.P(summary_hf)
        ]), html.Div([
            html.H3("Output from Falconsai:"),
            html.P(summary_custom)
        ]), html.Div([
            html.H3("แปลเป็นภาษาไทย (จาก Facebook/Bart):"),
            html.P(translated_text_hf)
        ]), html.Div([
            html.H3("แปลเป็นภาษาไทย (จาก Falconsai):"),
            html.P(translated_text_custom)
        ]), html.Div([
            html.H3("Sentence Similarity from Facebook/Bart:"),
            html.P(f"{similarity_hf:.2f}")
        ]), html.Div([
            html.H3("Sentence Similarity from Falconsai:"),
            html.P(f"{similarity_custom:.2f}")
        ]), {'data': data, 'layout': layout}, input_wordcloud_data, hf_wordcloud_data, custom_wordcloud_data
    else:
        print(datetime.now(), "update_output point 3, n_clicks=", n_clicks)
        return html.Div(), html.Div(), html.Div(), html.Div(), html.Div(), html.Div(), html.Div(), {}, html.Div(), html.Div(), html.Div()

print(datetime.now(), "step 3 finish")

print(datetime.now(), "step 4 run_server")

if __name__ == '__main__':
    app.run_server(debug=True)

print(datetime.now(), "step 4 finish")
