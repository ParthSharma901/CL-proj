import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import seaborn as sns
import json
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from googletrans import Translator
import warnings

warnings.filterwarnings('ignore')

# Try to download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class HinglishSentimentAnalyzer:
    def __init__(self):
        self.pos_neg_model = None
        self.serious_joke_model = None
        self.translator = Translator()
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))

        # Normalization dictionary for Hinglish
        self.normalization_dict = {
            # English shortenings
            "pls": "please", "plz": "please", "u": "you", "r": "are", "ur": "your",
            "thx": "thanks", "wud": "would", "wht": "what", "abt": "about",
            "bcoz": "because", "cuz": "because", "b4": "before", "gr8": "great",
            "btw": "by the way", "cya": "see you", "dm": "direct message",
            "idk": "i don't know", "lmk": "let me know", "tbh": "to be honest",

            # Hinglish specific corrections
            "kya": "kya", "hai": "hai", "nhi": "nahi", "kese": "kaise",
            "acha": "achha", "thik": "theek", "pyar": "pyaar", "ho": "ho",
            "kuch": "kuchh", "dost": "dost", "kam": "kaam", "jada": "zyada",
            "kaha": "kahaan", "tum": "tum", "me": "main", "mai": "main",
            "koi": "koi", "h": "hai", "k": "ke", "n": "ne", "ko": "ko",
            "ty": "thank you", "gn": "good night", "gm": "good morning"
        }

        # Dictionary of emoticons and their sentiments (positive/negative)
        self.emoticons = {
            ':)': 1.0, ':-)': 1.0, ':D': 1.5, ':-D': 1.5, ';)': 0.7, ';-)': 0.7,
            ':(': -1.0, ':-(': -1.0, ':/': -0.5, ':-/': -0.5, ':\'(': -1.5,
            ':-\'(': -1.5, ':P': 0.8, ':-P': 0.8, ':p': 0.8, ':-p': 0.8,
            ':|': 0, ':-|': 0, ':*': 1.0, ':-*': 1.0, '<3': 1.2
        }

        # Dictionary of sentiment words (both English and Hindi)
        self.sentiment_dict = {
            # English positive words
            'happy': 1.0, 'good': 0.8, 'great': 1.2, 'excellent': 1.5, 'amazing': 1.4,
            'wonderful': 1.3, 'awesome': 1.4, 'love': 1.2, 'beautiful': 1.1, 'best': 1.3,
            'better': 0.7, 'nice': 0.8, 'perfect': 1.5, 'fantastic': 1.4, 'pleasant': 0.9,
            'enjoy': 1.0, 'thanks': 0.8, 'thank': 0.8, 'glad': 0.9, 'exciting': 1.1,

            # English negative words
            'bad': -0.8, 'terrible': -1.4, 'awful': -1.3, 'horrible': -1.5, 'sad': -1.0,
            'hate': -1.3, 'dislike': -0.9, 'poor': -0.7, 'worst': -1.5, 'disappointed': -1.1,
            'disappointing': -1.0, 'unfortunate': -0.8, 'upset': -1.1, 'angry': -1.2, 'annoyed': -0.9,
            'sorry': -0.6, 'frustrating': -1.1, 'unhappy': -1.0, 'fail': -1.0, 'failure': -1.1,

            # Hindi positive words
            'achha': 0.8, 'accha': 0.8, 'badhiya': 1.0, 'shaanadaar': 1.3, 'uttam': 1.2,
            'pyaara': 1.0, 'sundar': 1.0, 'shandaar': 1.3, 'mast': 1.0, 'zabardast': 1.4,
            'bahut': 0.5, 'khushi': 1.1, 'khush': 1.0, 'pasand': 0.9, 'dhanayavaad': 0.8,
            'shukriya': 0.8, 'dhanyavaad': 0.8, 'shubhkamnaye': 0.9, 'umda': 1.0,

            # Hindi negative words
            'bura': -0.8, 'ganda': -0.9, 'kharaab': -0.8, 'bekaar': -0.8, 'dukhi': -1.0,
            'dard': -1.0, 'nafrat': -1.3, 'gussa': -1.1, 'naraz': -1.0, 'bekar': -0.8,
            'pareshan': -1.0, 'tenshun': -0.9, 'tension': -0.9, 'dukh': -1.1, 'rona': -0.9,
            'afsos': -0.8, 'buri': -0.8, 'galat': -0.7, 'naahi': -0.5, 'mat': -0.4,
        }

        # Dictionary for serious-jokeful dimension
        self.mood_dict = {
            # Serious words (English)
            'serious': -1.0, 'important': -0.8, 'critical': -1.0, 'urgent': -1.2,
            'crucial': -1.1, 'significant': -0.8, 'essential': -0.9, 'vital': -1.0,
            'grave': -1.3, 'severe': -1.1, 'emergency': -1.4, 'concern': -0.7,
            'careful': -0.6, 'caution': -0.8, 'warning': -0.9, 'attention': -0.6,

            # Jokeful words (English)
            'funny': 1.1, 'haha': 1.4, 'lol': 1.3, 'hilarious': 1.5, 'joke': 1.0,
            'humorous': 1.0, 'comedy': 1.1, 'amusing': 0.9, 'entertaining': 0.8,
            'laugh': 1.2, 'rofl': 1.5, 'lmao': 1.4, 'hehe': 1.2, 'giggle': 1.0,
            'kidding': 0.9, 'silly': 0.9, 'jk': 1.0, 'humor': 1.0, 'tease': 0.8,

            # Serious words (Hindi)
            'gambhir': -1.0, 'zaroori': -0.8, 'mahatvapurna': -0.9, 'zaruri': -0.8,
            'sankat': -1.2, 'chinta': -0.9, 'dhyan': -0.7, 'savdhan': -0.9,
            'chetavni': -1.0, 'chaukas': -0.8, 'sahi': -0.5, 'satark': -0.8,

            # Jokeful words (Hindi)
            'mazaak': 1.0, 'hasee': 1.1, 'mazak': 1.0, 'masti': 0.9, 'hasi': 1.1,
            'tamaasha': 0.8, 'tamasha': 0.8, 'mazedar': 0.9, 'haha': 1.4,
            'joke': 1.0, 'comedy': 1.1, 'hasna': 1.0, 'haso': 1.0
        }

        # Initialize transformer model for additional sentiment analysis
        try:
            print("Loading transformer model for sentiment analysis...")
            self.tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "nlptown/bert-base-multilingual-uncased-sentiment")
            self.sentiment_pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
            self.transformer_available = True
        except Exception as e:
            print(f"Transformer model could not be loaded: {e}")
            print("Continuing with dictionary-based approach only...")
            self.transformer_available = False

    def normalize_text(self, text):
        """Normalize Hinglish text by fixing spelling and removing extra spaces"""
        text = text.lower()
        tokens = text.split()
        normalized_tokens = [self.normalization_dict.get(token, token) for token in tokens]
        return ' '.join(normalized_tokens)

    def preprocess(self, text):
        """Preprocess Hinglish text for sentiment analysis"""
        # Normalize text
        text = self.normalize_text(text)

        # Remove URLs, mentions, and hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Join tokens back to text
        processed_text = ' '.join(tokens)

        return processed_text

    def analyze_sentiment(self, text):
        """Analyze sentiment of Hinglish text using rule-based approach"""
        # Preprocess text
        processed_text = self.preprocess(text)

        # Initialize sentiment scores
        pos_neg_score = 0
        serious_joke_score = 0

        # Split into words
        words = processed_text.lower().split()

        # Check for emoticons in original text (not processed)
        for emoticon, score in self.emoticons.items():
            if emoticon in text:
                pos_neg_score += score

        # Word-based sentiment analysis
        for word in words:
            # Positive-Negative dimension
            if word in self.sentiment_dict:
                pos_neg_score += self.sentiment_dict[word]

            # Serious-Jokeful dimension
            if word in self.mood_dict:
                serious_joke_score += self.mood_dict[word]

        # Normalize scores
        if len(words) > 0:
            divisor = max(1, min(len(words) / 3, 5))  # Normalize but don't over-dampen the effect
            pos_neg_score = pos_neg_score / divisor
            serious_joke_score = serious_joke_score / divisor

        # Clamp scores to [-1, 1]
        pos_neg_score = max(min(pos_neg_score, 1), -1)
        serious_joke_score = max(min(serious_joke_score, 1), -1)

        # If transformer model is available, enhance positive-negative score
        if self.transformer_available:
            try:
                # Try English version first
                en_result = self.sentiment_pipeline(processed_text)
                stars = int(en_result[0]['label'].split()[0])
                transformer_score = (stars - 3) / 2  # Convert 1-5 scale to -1 to 1

                # Blend dictionary and transformer scores
                pos_neg_score = (pos_neg_score + transformer_score) / 2
            except Exception as e:
                print(f"Transformer analysis failed: {e}")
                # Fallback to just dictionary score
                pass

        return pos_neg_score, serious_joke_score

    def analyze_multiple(self, sentences):
        """Analyze multiple sentences and return scores for each"""
        results = []
        for sentence in sentences:
            pos_neg, serious_joke = self.analyze_sentiment(sentence)
            results.append({
                'text': sentence,
                'positive_negative': pos_neg,
                'serious_jokeful': serious_joke,
                'quadrant': self._determine_quadrant(pos_neg, serious_joke)
            })

        return results

    def _determine_quadrant(self, pos_neg, serious_joke):
        """Determine which quadrant the sentiment falls into"""
        if pos_neg >= 0 and serious_joke >= 0:
            return "Positive and Jokeful"
        elif pos_neg >= 0 and serious_joke < 0:
            return "Positive and Serious"
        elif pos_neg < 0 and serious_joke >= 0:
            return "Negative and Jokeful"
        else:
            return "Negative and Serious"

    def plot_sentiment_quadrant(self, results, save_path=None):
        """Plot sentiment analysis results on a 4-quadrant graph"""
        # Extract scores from results
        pos_neg_scores = [r['positive_negative'] for r in results]
        serious_joke_scores = [r['serious_jokeful'] for r in results]
        texts = [r['text'] for r in results]

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create scatter plot with points colored by density
        scatter = ax.scatter(pos_neg_scores, serious_joke_scores,
                             alpha=0.7, s=100, c=range(len(pos_neg_scores)),
                             cmap='viridis', edgecolors='w')

        # Add labels for points
        for i, txt in enumerate(texts):
            # Truncate long texts
            short_txt = txt[:20] + "..." if len(txt) > 20 else txt
            ax.annotate(short_txt, (pos_neg_scores[i], serious_joke_scores[i]),
                        fontsize=8, ha='center', va='bottom',
                        xytext=(0, 5), textcoords='offset points')

        # Add quadrant labels
        ax.text(0.5, 0.5, "Positive & Jokeful", fontsize=12, ha='center', va='center', alpha=0.7)
        ax.text(0.5, -0.5, "Positive & Serious", fontsize=12, ha='center', va='center', alpha=0.7)
        ax.text(-0.5, 0.5, "Negative & Jokeful", fontsize=12, ha='center', va='center', alpha=0.7)
        ax.text(-0.5, -0.5, "Negative & Serious", fontsize=12, ha='center', va='center', alpha=0.7)

        # Add quadrant dividers
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

        # Set limits and labels
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel('Negative (-1) to Positive (+1)', fontsize=14)
        ax.set_ylabel('Serious (-1) to Jokeful (+1)', fontsize=14)
        ax.set_title('Hinglish Sentiment Analysis: 4-Quadrant Emotion Map', fontsize=16)

        # Add colorbar to show order of sentences
        cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', pad=0.01)
        cbar.set_label('Sentence Order', rotation=270, labelpad=15)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)

        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.tight_layout()

        return fig, ax


def read_sentences_from_file(input_file):
    """Read sentences from a text file, one per line"""
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        # Use default sentences if file not found
        return [
            "mera din bahut achha tha",
            "kya bakwaas hai yeh",
            "mujhe yeh joke bahut funny laga",
            "ye exam bahut mushkil hai",
            "aaj main bahut khush hu because mera birthday hai"
        ]

    with open(input_file, 'r', encoding='utf-8') as file:
        sentences = [line.strip() for line in file if line.strip()]

    return sentences


def write_results_to_file(results, output_file):
    """Write sentiment analysis results to a text file"""
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("Hinglish Sentiment Analysis Results\n")
        file.write("================================\n\n")

        for i, result in enumerate(results):
            file.write(f"{i + 1}. Text: {result['text']}\n")
            file.write(f"   Positive/Negative Score: {result['positive_negative']:.2f}\n")
            file.write(f"   Serious/Jokeful Score: {result['serious_jokeful']:.2f}\n")
            file.write(f"   Quadrant: {result['quadrant']}\n\n")


def save_results_as_json(results, json_file):
    """Save sentiment analysis results as JSON for potential further processing"""
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=2)


def main():
    # Define file paths
    input_file = "hinglish_sentences.txt"
    output_file = "sentiment_analysis_results.txt"
    json_output = "sentiment_analysis_results.json"
    plot_file = "hinglish_sentiment_quadrant.png"

    # Create analyzer instance
    analyzer = HinglishSentimentAnalyzer()

    # Read sentences from file
    print(f"Reading sentences from {input_file}...")
    sentences = read_sentences_from_file(input_file)
    print(f"Found {len(sentences)} sentences to analyze.")

    # Analyze sentences
    print("Analyzing sentences...")
    results = analyzer.analyze_multiple(sentences)

    # Write results to output file
    write_results_to_file(results, output_file)
    print(f"Analysis results written to {output_file}")

    # Save results as JSON for potential further processing
    save_results_as_json(results, json_output)
    print(f"Results also saved as JSON at {json_output}")

    # Plot results
    print("Generating sentiment quadrant plot...")
    analyzer.plot_sentiment_quadrant(results, save_path=plot_file)

    print(f"Analysis complete! Plot has been saved to {plot_file}")


if __name__ == "__main__":
    main()