from flask import Flask, render_template, request
from transformers import pipeline
from datasets import load_dataset
from PIL import Image
import re

app = Flask(__name__)

def preprocess(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    return text

def summarize(text):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", framework="pt")
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            article = uploaded_file.read().decode("utf-8")
            article = preprocess(article)

            summary = summarize(article)

            return render_template('summary.html', article=article, summary=summary)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
