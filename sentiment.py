import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import seaborn as sns
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
import warnings

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('vader_lexicon')
    nltk.data.find('averaged_perceptron_tagger')
    nltk.data.find('sentiwordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('sentiwordnet')


def polarity(a, b):
    """Calculate the SentiWordNet polarity of a word with its part of speech tag"""
    score = 0
    # FIX: Convert filter objects to list before checking length
    if ((b == 'VB')):
        synsets = list(swn.senti_synsets(a, 'v'))
        if len(synsets) > 0:
            for i in synsets:
                score += (i).pos_score() - (i).neg_score()
            score = score / len(synsets)
            return score

    if ((b == 'NN')):
        synsets = list(swn.senti_synsets(a, 'n'))
        if len(synsets) > 0:
            for i in synsets:
                score += (i).pos_score() - (i).neg_score()
            score = score / len(synsets)
            return score

    if ((b == 'JJ')):
        synsets = list(swn.senti_synsets(a, 'a'))
        if len(synsets) > 0:
            for i in synsets:
                score += (i).pos_score() - (i).neg_score()
            score = score / len(synsets)
            return score

    if ((b == 'RB')):
        synsets = list(swn.senti_synsets(a, 'r'))
        if len(synsets) > 0:
            for i in synsets:
                score += (i).pos_score() - (i).neg_score()
            score = score / len(synsets)
            return score
    if (score == 0):
        return 'NF'


class EnhancedHinglishSentimentAnalyzer:
    def __init__(self):
        self.pos_neg_model = None
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))

        # Initialize NLTK's Vader sentiment analyzer
        self.vader = SentimentIntensityAnalyzer()

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
            'joy': 1.1, 'delighted': 1.2, 'pleased': 0.9, 'satisfied': 0.8, 'impressed': 1.0,
            'favorite': 0.9, 'brilliant': 1.3, 'superb': 1.4, 'outstanding': 1.4, 'incredible': 1.3,
            'fabulous': 1.3, 'splendid': 1.2, 'terrific': 1.2, 'marvelous': 1.3, 'wow': 1.5,
            'right': 0.5, 'please': 0.3, 'ok': 0.4, 'okay': 0.4,

            # English negative words
            'bad': -0.8, 'terrible': -1.4, 'awful': -1.3, 'horrible': -1.5, 'sad': -1.0,
            'hate': -1.3, 'dislike': -0.9, 'poor': -0.7, 'worst': -1.5, 'disappointed': -1.1,
            'disappointing': -1.0, 'unfortunate': -0.8, 'upset': -1.1, 'angry': -1.2, 'annoyed': -0.9,
            'sorry': -0.6, 'frustrating': -1.1, 'unhappy': -1.0, 'fail': -1.0, 'failure': -1.1,
            'terrible': -1.3, 'miserable': -1.2, 'disgusting': -1.4, 'pathetic': -1.3, 'useless': -1.0,
            'annoying': -1.0, 'rubbish': -1.1, 'crap': -1.2, 'boring': -0.7, 'lousy': -0.9,
            'dreadful': -1.3, 'offensive': -1.1, 'unprofessional': -0.8, 'disappointed': -0.9,

            # Hindi positive words
            'achha': 0.8, 'accha': 0.8, 'badhiya': 1.0, 'shaanadaar': 1.3, 'uttam': 1.2,
            'pyaara': 1.0, 'sundar': 1.0, 'shandaar': 1.3, 'mast': 1.0, 'zabardast': 1.4,
            'bahut': 0.5, 'khushi': 1.1, 'khush': 1.0, 'pasand': 0.9, 'dhanayavaad': 0.8,
            'shukriya': 0.8, 'dhanyavaad': 0.8, 'shubhkamnaye': 0.9, 'umda': 1.0,
            'behtareen': 1.3, 'kamal': 1.1, 'aala': 1.0, 'anand': 1.0, 'sukoon': 0.9,
            'mehrbani': 0.7, 'khoobsurat': 1.2, 'pyaar': 1.2, 'prem': 1.0, 'shanti': 0.8,
            'prasannata': 1.0, 'sukhi': 1.0, 'harsha': 1.0, 'priya': 0.9, 'divya': 1.1,
            'sahi': 0.7, 'theek': 0.5, 'haan': 0.4,

            # Hindi negative words
            'bura': -0.8, 'ganda': -0.9, 'kharaab': -0.8, 'bekaar': -0.8, 'dukhi': -1.0,
            'dard': -1.0, 'nafrat': -1.3, 'gussa': -1.1, 'naraz': -1.0, 'bekar': -0.8,
            'pareshan': -1.0, 'tenshun': -0.9, 'tension': -0.9, 'dukh': -1.1, 'rona': -0.9,
            'afsos': -0.8, 'buri': -0.8, 'galat': -0.7, 'naahi': -0.5, 'mat': -0.4,
            'ghatiya': -1.2, 'bakwas': -1.1, 'faltu': -0.8, 'befaltu': -0.9, 'nakhush': -1.0,
            'peeda': -1.0, 'chinta': -0.8, 'krodh': -1.2, 'kharab': -0.9, 'doshi': -0.7,
            'dhokha': -1.2, 'shikayat': -0.7, 'bimari': -1.0, 'takleef': -1.1, 'mushkil': -0.7,
        }

        # Initialize transformer models for additional sentiment analysis
        try:
            print("Loading transformer model for sentiment analysis...")
            # Multilingual sentiment model
            self.tokenizer_sentiment = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
            self.model_sentiment = AutoModelForSequenceClassification.from_pretrained(
                "nlptown/bert-base-multilingual-uncased-sentiment")
            self.sentiment_pipeline = pipeline("sentiment-analysis", model=self.model_sentiment,
                                               tokenizer=self.tokenizer_sentiment)

            self.transformer_available = True
            print("Transformer model loaded successfully!")
        except Exception as e:
            print(f"Transformer model could not be loaded: {e}")
            print("Continuing with dictionary-based approach only...")
            self.transformer_available = False

    def preprocess(self, text):
        """
        Preprocessing is minimal since we're using already normalized text
        from hinglish_norm.py, but we still do some basic cleanup
        """
        # Remove URLs, mentions, and hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Join tokens back to text
        processed_text = ' '.join(tokens)

        return processed_text, tokens

    def get_sentiwordnet_scores(self, tokens):
        """Get SentiWordNet scores for tokens with appropriate POS tags"""
        pos_tags = nltk.pos_tag(tokens)
        swn_scores = []

        for word, pos in pos_tags:
            # Map NLTK POS tags to SentiWordNet format
            pos_prefix = pos[:2]
            if pos_prefix in ['NN', 'JJ', 'RB', 'VB']:
                score = polarity(word.lower(), pos_prefix)
                if score != 'NF':
                    swn_scores.append(score)

        return swn_scores

    def analyze_sentiment(self, text):
        """
        Analyze sentiment of Hinglish text using multiple approaches and combine them
        Focus only on positive/negative dimension
        """
        # Preprocess text and get tokens
        processed_text, tokens = self.preprocess(text)

        # Initialize sentiment score
        pos_neg_score = 0

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

        # Use NLTK's VADER for additional sentiment analysis
        vader_scores = self.vader.polarity_scores(processed_text)
        vader_compound = vader_scores['compound']  # Range: -1 to 1

        # Use TextBlob for additional sentiment perspective
        textblob_polarity = TextBlob(processed_text).sentiment.polarity  # Range: -1 to 1

        # Get SentiWordNet scores (NEW!)
        swn_scores = self.get_sentiwordnet_scores(tokens)
        swn_score = sum(swn_scores) / max(1, len(swn_scores)) if swn_scores else 0

        # Combine all scores (including SentiWordNet)
        if swn_scores:
            pos_neg_score = (pos_neg_score + vader_compound + textblob_polarity + swn_score) / 4
        else:
            pos_neg_score = (pos_neg_score + vader_compound + textblob_polarity) / 3

        # Normalize scores based on text length
        if len(words) > 0:
            divisor = max(1, min(len(words) / 3, 5))  # Normalize but don't over-dampen the effect
            pos_neg_score = pos_neg_score / divisor

        # Clamp scores to [-1, 1]
        pos_neg_score = max(min(pos_neg_score, 1), -1)

        # If transformer model is available, enhance scores
        transformer_score = 0
        if self.transformer_available:
            try:
                # Get sentiment from transformer model
                en_result = self.sentiment_pipeline(processed_text)

                # Handle the transformer output format safely
                if isinstance(en_result, list) and len(en_result) > 0:
                    # Try to extract sentiment score safely from the result
                    result_item = en_result[0]
                    if isinstance(result_item, dict):
                        label = result_item.get('label', '')

                        # Try to extract a score from the label
                        if isinstance(label, str) and label:
                            # Extract the first number from the label if available
                            import re
                            numbers = re.findall(r'\d+', label)
                            if numbers:
                                star_rating = int(numbers[0])
                                transformer_score = (star_rating - 3) / 2  # Convert to -1 to 1 scale
                            else:
                                # No numbers found, use the sentiment mapping
                                sentiment_mapping = {
                                    'positive': 0.7,
                                    'negative': -0.7,
                                    'neutral': 0,
                                    '1 star': -1.0,
                                    '2 stars': -0.5,
                                    '3 stars': 0,
                                    '4 stars': 0.5,
                                    '5 stars': 1.0
                                }
                                transformer_score = sentiment_mapping.get(label.lower(), 0)

                # Blend dictionary-based score with transformer score
                if transformer_score != 0:  # Only blend if we got a valid transformer score
                    pos_neg_score = (pos_neg_score * 0.7) + (transformer_score * 0.3)

            except Exception as e:
                print(f"Transformer analysis failed: {str(e)}")
                # Fallback to just dictionary score
                pass

        return pos_neg_score

    def analyze_multiple(self, sentences):
        """Analyze multiple sentences and return scores for each"""
        results = []
        for sentence in sentences:
            sentiment_score = self.analyze_sentiment(sentence)
            sentiment_category = "Positive" if sentiment_score >= 0 else "Negative"
            sentiment_strength = abs(sentiment_score)

            results.append({
                'text': sentence,
                'sentiment_score': sentiment_score,
                'sentiment_category': sentiment_category,
                'sentiment_strength': sentiment_strength
            })

        return results

    def plot_sentiment_histogram(self, results, save_path=None):
        """
        Plot sentiment analysis results as a horizontal bar chart
        with positive/negative sentiment on x-axis and sentences on y-axis
        """
        # Extract data from results
        texts = [r['text'][:30] + '...' if len(r['text']) > 30 else r['text'] for r in results]
        sentiment_scores = [r['sentiment_score'] for r in results]

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(10, max(6, len(texts) * 0.5)))

        # Create horizontal bar chart
        bars = ax.barh(
            range(len(texts)),
            sentiment_scores,
            height=0.7,
            color=[plt.cm.RdYlGn(0.5 * (score + 1)) for score in sentiment_scores]
        )

        # Add a vertical line at x=0
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)

        # Add labels and title
        ax.set_yticks(range(len(texts)))
        ax.set_yticklabels(texts)
        ax.set_xlabel('Negative (-1) to Positive (+1)', fontsize=12)
        ax.set_title('Hinglish Sentiment Analysis with SentiWordNet', fontsize=14)

        # Set x-axis limits
        ax.set_xlim(-1.1, 1.1)

        # Add grid lines
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            score = sentiment_scores[i]
            label_position = 0.05 if score < 0 else -0.05
            alignment = 'left' if score < 0 else 'right'
            ax.text(
                score + label_position,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}",
                ha=alignment,
                va='center',
                color='black',
                fontweight='bold'
            )

        # Add color legend
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(-1, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1)
        cbar.set_label('Sentiment Score')

        plt.tight_layout()

        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        return fig, ax


def read_normalized_sentences(norm_file='normalized_hinglish.txt'):
    """
    Read normalized sentences from the output file of hinglish_norm.py
    Format expected:
    Original [1]: original text
    Normalized [1]: normalized text

    Original [2]: original text
    Normalized [2]: normalized text
    """
    normalized_sentences = []

    if not os.path.exists(norm_file):
        print(f"Error: Normalized file {norm_file} not found!")
        print("Please run hinglish_norm.py first to generate the normalized text.")
        return []

    with open(norm_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if line.strip().startswith("Normalized ["):
            # Extract normalized text after the colon
            parts = line.split(":", 1)
            if len(parts) > 1:
                normalized_text = parts[1].strip()
                normalized_sentences.append(normalized_text)

    return normalized_sentences


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
        file.write("Hinglish Sentiment Analysis Results with SentiWordNet\n")
        file.write("==============================================\n\n")

        for i, result in enumerate(results):
            file.write(f"{i + 1}. Text: {result['text']}\n")
            file.write(f"   Sentiment Score: {result['sentiment_score']:.2f}\n")
            file.write(
                f"   Category: {result['sentiment_category']} (Strength: {result['sentiment_strength']:.2f})\n\n")


def save_results_as_json(results, json_file):
    """Save sentiment analysis results as JSON for potential further processing"""
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=2)


def main():
    # Define file paths
    normalized_file = "normalized_hinglish.txt"  # Output from hinglish_norm.py
    input_file = "hinglish_sentences.txt"  # Original input, used as fallback
    output_file = "sentiment_analysis_results.txt"
    json_output = "sentiment_analysis_results.json"
    histogram_file = "hinglish_sentiment_histogram.png"

    # Create analyzer instance
    analyzer = EnhancedHinglishSentimentAnalyzer()

    # Try to read normalized sentences first, fall back to original if needed
    print(f"Reading normalized sentences from {normalized_file}...")
    sentences = read_normalized_sentences(normalized_file)

    if not sentences:
        print(f"Falling back to reading original sentences from {input_file}...")
        sentences = read_sentences_from_file(input_file)

    print(f"Found {len(sentences)} sentences to analyze.")

    # Analyze sentences
    print("Analyzing sentences with SentiWordNet enhancement...")
    results = analyzer.analyze_multiple(sentences)

    # Write results to output file
    write_results_to_file(results, output_file)
    print(f"Analysis results written to {output_file}")

    # Save results as JSON for potential further processing
    save_results_as_json(results, json_output)
    print(f"Results also saved as JSON at {json_output}")

    # Create only the histogram visualization
    print("Generating sentiment histogram...")
    analyzer.plot_sentiment_histogram(results, save_path=histogram_file)

    print(f"Analysis complete! Histogram saved to: {histogram_file}")


if __name__ == "__main__":
    main()