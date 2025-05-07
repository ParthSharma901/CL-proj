import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import seaborn as sns
import json
from textblob import TextBlob
import warnings

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('vader_lexicon')
    nltk.data.find('averaged_perceptron_tagger')
    nltk.data.find('sentiwordnet')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('sentiwordnet')
    nltk.download('wordnet')


def penn_to_wn(tag):
    """Convert Penn Treebank tags to WordNet tags"""
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


def get_wordnet_pos(word):
    """Get the part-of-speech tag for a word"""
    tag = nltk.pos_tag([word])[0][1]
    tag = penn_to_wn(tag)
    return tag


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


class HinglishSentimentAnalyzer:
    def __init__(self):
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

            # Hindi positive words (Latin script)
            'achha': 0.8, 'accha': 0.8, 'badhiya': 1.0, 'shaanadaar': 1.3, 'uttam': 1.2,
            'pyaara': 1.0, 'sundar': 1.0, 'shandaar': 1.3, 'mast': 1.0, 'zabardast': 1.4,
            'bahut': 0.5, 'khushi': 1.1, 'khush': 1.0, 'pasand': 0.9, 'dhanayavaad': 0.8,
            'shukriya': 0.8, 'dhanyavaad': 0.8, 'shubhkamnaye': 0.9, 'umda': 1.0,
            'behtareen': 1.3, 'kamal': 1.1, 'aala': 1.0, 'anand': 1.0, 'sukoon': 0.9,
            'mehrbani': 0.7, 'khoobsurat': 1.2, 'pyaar': 1.2, 'prem': 1.0, 'shanti': 0.8,
            'prasannata': 1.0, 'sukhi': 1.0, 'harsha': 1.0, 'priya': 0.9, 'divya': 1.1,
            'sahi': 0.7, 'theek': 0.5, 'haan': 0.4,

            # Hindi negative words (Latin script)
            'bura': -0.8, 'ganda': -0.9, 'kharaab': -0.8, 'bekaar': -0.8, 'dukhi': -1.0,
            'dard': -1.0, 'nafrat': -1.3, 'gussa': -1.1, 'naraz': -1.0, 'bekar': -0.8,
            'pareshan': -1.0, 'tenshun': -0.9, 'tension': -0.9, 'dukh': -1.1, 'rona': -0.9,
            'afsos': -0.8, 'buri': -0.8, 'galat': -0.7, 'naahi': -0.5, 'mat': -0.4,
            'ghatiya': -1.2, 'bakwas': -1.1, 'faltu': -0.8, 'befaltu': -0.9, 'nakhush': -1.0,
            'peeda': -1.0, 'chinta': -0.8, 'krodh': -1.2, 'kharab': -0.9, 'doshi': -0.7,
            'dhokha': -1.2, 'shikayat': -0.7, 'bimari': -1.0, 'takleef': -1.1, 'mushkil': -0.7,

            # Hindi positive words (Devanagari script)
            'अच्छा': 0.8, 'बढ़िया': 1.0, 'शानदार': 1.3, 'उत्तम': 1.2,
            'प्यारा': 1.0, 'सुंदर': 1.0, 'मस्त': 1.0, 'जबरदस्त': 1.4,
            'बहुत': 0.5, 'खुशी': 1.1, 'खुश': 1.0, 'पसंद': 0.9, 'धन्यवाद': 0.8,
            'शुक्रिया': 0.8, 'शुभकामनाएं': 0.9, 'उम्दा': 1.0,
            'बेहतरीन': 1.3, 'कमाल': 1.1, 'आला': 1.0, 'आनंद': 1.0, 'सुकून': 0.9,
            'मेहरबानी': 0.7, 'खूबसूरत': 1.2, 'प्यार': 1.2, 'प्रेम': 1.0, 'शांति': 0.8,
            'प्रसन्नता': 1.0, 'सुखी': 1.0, 'हर्ष': 1.0, 'प्रिय': 0.9, 'दिव्य': 1.1,
            'सही': 0.7, 'ठीक': 0.5, 'हाँ': 0.4,

            # Hindi negative words (Devanagari script)
            'बुरा': -0.8, 'गंदा': -0.9, 'खराब': -0.8, 'बेकार': -0.8, 'दुखी': -1.0,
            'दर्द': -1.0, 'नफरत': -1.3, 'गुस्सा': -1.1, 'नाराज़': -1.0,
            'परेशान': -1.0, 'टेंशन': -0.9, 'दुःख': -1.1, 'रोना': -0.9,
            'अफसोस': -0.8, 'बुरी': -0.8, 'गलत': -0.7, 'नहीं': -0.5, 'मत': -0.4,
            'घटिया': -1.2, 'बकवास': -1.1, 'फालतू': -0.8, 'नाखुश': -1.0,
            'पीड़ा': -1.0, 'चिंता': -0.8, 'क्रोध': -1.2, 'दोषी': -0.7,
            'धोखा': -1.2, 'शिकायत': -0.7, 'बीमारी': -1.0, 'तकलीफ': -1.1, 'मुश्किल': -0.7,
        }

        # Neutral threshold for reclassification
        self.neutral_threshold = 0.2

    def preprocess(self, text):
        """
        Preprocessing for mixed English-Devanagari text
        """
        # Remove URLs, mentions, and hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)

        # For mixed scripts, we need to handle tokenization carefully
        tokens = word_tokenize(text)

        return text, tokens

    def get_sentiwordnet_scores(self, tokens):
        """Get SentiWordNet scores for tokens with appropriate POS tags"""
        # Filter out Devanagari tokens (only apply to English tokens)
        english_tokens = [token for token in tokens if all(ord(char) < 2304 for char in token)]

        if not english_tokens:
            return []

        pos_tags = nltk.pos_tag(english_tokens)
        swn_scores = []

        for word, pos in pos_tags:
            # Map NLTK POS tags to SentiWordNet format
            pos_prefix = pos[:2]
            if pos_prefix in ['NN', 'JJ', 'RB', 'VB']:
                score = polarity(word.lower(), pos_prefix)
                if score != 'NF':
                    swn_scores.append(score)

        return swn_scores

    def reclassify_with_wordnet(self, text, base_score):
        """
        Reclassify neutral sentiments using WordNet for deeper analysis
        """
        if abs(base_score) >= self.neutral_threshold:
            # If already clearly positive or negative, no need to reclassify
            return base_score

        # For neutral texts, do a deeper analysis
        english_words = [word for word in text.split() if all(ord(char) < 2304 for char in word)]
        if not english_words:
            return base_score

        wordnet_score = 0
        count = 0

        for word in english_words:
            # Skip very short words
            if len(word) < 3:
                continue

            # Get part of speech tag
            pos = get_wordnet_pos(word)
            if pos is None:
                continue

            # Find synsets
            synsets = list(wn.synsets(word, pos=pos))
            if not synsets:
                continue

            # Check sentiment from SentiWordNet
            for synset in synsets:
                swn_synset = swn.senti_synset(synset.name())
                if swn_synset:
                    wordnet_score += swn_synset.pos_score() - swn_synset.neg_score()
                    count += 1

        # Calculate average and return the enhanced score
        if count > 0:
            wordnet_score = wordnet_score / count
            # Blend the original score with the WordNet score
            return (base_score * 0.4) + (wordnet_score * 0.6)
        else:
            return base_score

    def analyze_sentiment(self, text):
        """
        Analyze sentiment of mixed English-Devanagari text using lexicon-based approach
        with WordNet reclassification for neutral texts
        """
        # Preprocess text and get tokens
        processed_text, tokens = self.preprocess(text)

        # Initialize sentiment score
        lexicon_score = 0

        # Check for emoticons in original text
        for emoticon, score in self.emoticons.items():
            if emoticon in text:
                lexicon_score += score

        # Word-based sentiment analysis using our lexicon
        for word in tokens:
            word_lower = word.lower()
            if word_lower in self.sentiment_dict:
                lexicon_score += self.sentiment_dict[word_lower]

        # Use VADER for additional English sentiment analysis
        vader_scores = self.vader.polarity_scores(processed_text)
        vader_compound = vader_scores['compound']  # Range: -1 to 1

        # Calculate average score from lexicon and VADER
        if len(tokens) > 0:
            divisor = max(1, min(len(tokens) / 3, 5))  # Normalize for text length
            lexicon_score = lexicon_score / divisor

        # Combine lexicon score with VADER score
        base_score = (lexicon_score * 0.7) + (vader_compound * 0.3)
        base_score = max(min(base_score, 1), -1)  # Clamp to [-1, 1]

        # Check if score is near neutral and needs reclassification
        if abs(base_score) < self.neutral_threshold:
            final_score = self.reclassify_with_wordnet(processed_text, base_score)
        else:
            final_score = base_score

        # Determine sentiment category
        if final_score >= self.neutral_threshold:
            sentiment_category = "Positive"
        elif final_score <= -self.neutral_threshold:
            sentiment_category = "Negative"
        else:
            sentiment_category = "Neutral"

        return {
            'score': final_score,
            'category': sentiment_category,
            'strength': abs(final_score)
        }

    def analyze_multiple(self, sentences):
        """Analyze multiple sentences and return scores for each"""
        results = []
        for sentence in sentences:
            sentiment_result = self.analyze_sentiment(sentence)
            results.append({
                'text': sentence,
                'sentiment_score': sentiment_result['score'],
                'sentiment_category': sentiment_result['category'],
                'sentiment_strength': sentiment_result['strength']
            })

        return results

    def plot_sentiment_histogram(self, results, save_path=None):
        """
        Plot sentiment analysis results as a horizontal bar chart
        with positive/negative/neutral sentiment on x-axis and sentences on y-axis
        """
        # Extract data from results
        texts = [r['text'][:30] + '...' if len(r['text']) > 30 else r['text'] for r in results]
        sentiment_scores = [r['sentiment_score'] for r in results]
        categories = [r['sentiment_category'] for r in results]

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(10, max(6, len(texts) * 0.5)))

        # Create horizontal bar chart with colored bars based on sentiment category
        colors = {
            'Positive': '#4CAF50',  # Green
            'Negative': '#F44336',  # Red
            'Neutral': '#9E9E9E'  # Gray
        }

        bar_colors = [colors[category] for category in categories]

        bars = ax.barh(
            range(len(texts)),
            sentiment_scores,
            height=0.7,
            color=bar_colors
        )

        # Add a vertical line at x=0
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)

        # Add labels and title
        ax.set_yticks(range(len(texts)))
        ax.set_yticklabels(texts)
        ax.set_xlabel('Sentiment Score: Negative (-1) to Neutral (0) to Positive (+1)', fontsize=12)
        ax.set_title('Hinglish-Devanagari Sentiment Analysis', fontsize=14)

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
                f"{score:.2f} ({categories[i]})",
                ha=alignment,
                va='center',
                color='black',
                fontweight='bold',
                fontsize=9
            )

        # Add legend for sentiment categories
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color=colors['Positive']),
            plt.Rectangle((0, 0), 1, 1, color=colors['Neutral']),
            plt.Rectangle((0, 0), 1, 1, color=colors['Negative'])
        ]
        ax.legend(legend_handles, ['Positive', 'Neutral', 'Negative'],
                  loc='lower right', frameon=True)

        plt.tight_layout()

        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        return fig, ax


def read_mixed_sentences(devanagari_file='devanagari_output.txt'):
    """
    Read mixed English-Devanagari sentences from our output file
    Format expected:
    Original [1]: original text
    Normalized [1]: normalized text
    Mixed E-H [1]: mixed text with Devanagari

    Original [2]: original text
    ...
    """
    mixed_sentences = []

    if not os.path.exists(devanagari_file):
        print(f"Error: Mixed text file {devanagari_file} not found!")
        print("Please run hinglish_norm.py first to generate the mixed English-Devanagari text.")
        return []

    with open(devanagari_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if line.strip().startswith("Mixed E-H ["):
            # Extract mixed text after the colon
            parts = line.split(":", 1)
            if len(parts) > 1:
                mixed_text = parts[1].strip()
                mixed_sentences.append(mixed_text)

    return mixed_sentences


def read_sentences_from_file(input_file):
    """Read sentences from a text file, one per line (fallback)"""
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        # Use default sentences if file not found
        return [
            "मेरा din bahut अच्छा था",
            "kya बकवास hai yeh",
            "mujhe yeh joke bahut फनी laga",
            "ye exam bahut मुश्किल hai",
            "aaj main bahut खुश hu because मेरा birthday hai"
        ]

    with open(input_file, 'r', encoding='utf-8') as file:
        sentences = [line.strip() for line in file if line.strip()]

    return sentences


def write_results_to_file(results, output_file):
    """Write sentiment analysis results to a text file"""
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("Hinglish-Devanagari Sentiment Analysis Results\n")
        file.write("===========================================\n\n")

        for i, result in enumerate(results):
            file.write(f"{i + 1}. Text: {result['text']}\n")
            file.write(f"   Sentiment Score: {result['sentiment_score']:.2f}\n")
            file.write(
                f"   Category: {result['sentiment_category']} (Strength: {result['sentiment_strength']:.2f})\n\n")


def save_results_as_json(results, json_file):
    """Save sentiment analysis results as JSON for potential further processing"""
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


def main():
    # Define file paths
    devanagari_file = "devanagari_output.txt"  # Output from hinglish_norm.py with mixed E-H
    input_file = "hinglish_sentences.txt"  # Original input, used as fallback
    output_file = "sentiment_analysis_results.txt"
    json_output = "sentiment_analysis_results.json"
    histogram_file = "hinglish_sentiment_histogram.png"

    # Create analyzer instance
    analyzer = HinglishSentimentAnalyzer()

    # Read mixed English-Devanagari sentences
    print(f"Reading mixed English-Devanagari sentences from {devanagari_file}...")
    sentences = read_mixed_sentences(devanagari_file)

    if not sentences:
        print(f"Falling back to reading original sentences from {input_file}...")
        sentences = read_sentences_from_file(input_file)

    print(f"Found {len(sentences)} sentences to analyze.")

    # Analyze sentences
    print("Analyzing mixed English-Devanagari sentences...")
    results = analyzer.analyze_multiple(sentences)

    # Write results to output file
    write_results_to_file(results, output_file)
    print(f"Analysis results written to {output_file}")

    # Save results as JSON for potential further processing
    save_results_as_json(results, json_output)
    print(f"Results also saved as JSON at {json_output}")

    # Create histogram visualization
    print("Generating sentiment histogram...")
    analyzer.plot_sentiment_histogram(results, save_path=histogram_file)

    print(f"Analysis complete! Histogram saved to: {histogram_file}")

    # Display summary of results
    categories = [r['sentiment_category'] for r in results]
    pos_count = categories.count('Positive')
    neg_count = categories.count('Negative')
    neu_count = categories.count('Neutral')

    print("\nSummary of Results:")
    print(f"Total sentences analyzed: {len(results)}")
    print(f"Positive: {pos_count} ({pos_count / len(results) * 100:.1f}%)")
    print(f"Negative: {neg_count} ({neg_count / len(results) * 100:.1f}%)")
    print(f"Neutral: {neu_count} ({neu_count / len(results) * 100:.1f}%)")


if __name__ == "__main__":
    main()