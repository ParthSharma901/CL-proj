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
import platform
import json
from textblob import TextBlob
import warnings
import emoji
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import unicodedata
import platform

warnings.filterwarnings('ignore')

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
    tag = nltk.pos_tag([word])[0][1]
    tag = penn_to_wn(tag)
    return tag


def polarity(a, b):
    score = 0
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

        self.vader = SentimentIntensityAnalyzer()
        self.setup_fonts()

        # Dictionary of emoticons and their sentiments (positive/negative)
        self.emoticons = {
            ':)': 1.0, ':-)': 1.0, ':D': 1.5, ':-D': 1.5, ';)': 0.7, ';-)': 0.7,
            ':(': -1.0, ':-(': -1.0, ':/': -0.5, ':-/': -0.5, ':\'(': -1.5,
            ':-\'(': -1.5, ':P': 0.8, ':-P': 0.8, ':p': 0.8, ':-p': 0.8,
            ':|': 0, ':-|': 0, ':*': 1.0, ':-*': 1.0, '<3': 1.2
        }

        # Dictionary of emojis and their sentiment weights
        self.emoji_sentiment = {
            # Positive emojis
            '😊': 1.0,  # Smiling face with smiling eyes
            '😄': 1.2,  # Grinning face with smiling eyes
            '😁': 1.1,  # Beaming face with smiling eyes
            '😍': 1.5,  # Smiling face with heart-eyes
            '🥰': 1.4,  # Smiling face with hearts
            '👍': 0.8,  # Thumbs up
            '❤️': 1.3,  # Red heart
            '🎉': 1.0,  # Party popper
            '✨': 0.7,  # Sparkles
            '🙏': 0.9,  # Folded hands (thank you/please)
            '🤗': 1.0,  # Hugging face

            # Negative emojis
            '😢': -1.0,  # Crying face
            '😭': -1.3,  # Loudly crying face
            '😞': -0.8,  # Disappointed face
            '😔': -0.7,  # Pensive face
            '😡': -1.4,  # Pouting face (angry)
            '👎': -0.8,  # Thumbs down
            '😠': -1.2,  # Angry face
            '😒': -0.9,  # Unamused face
            '😩': -1.1,  # Weary face
            '😫': -1.2,  # Tired face
            '🙄': -0.6,  # Face with rolling eyes
            '😱':-0.4,  #scared emoji
            '☠️': -0.5, #skull emoji
            '🤡':-1.3, # clown emoji

            # Neutral/ambiguous emojis
            '😐': 0.0,  # Neutral face
            '🤔': -0.1  # Thinking face (slightly negative)
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

    def setup_fonts(self):
        # Determine the best font to use based on operating system
        self.best_font = self.get_unicode_font()

        # Configure matplotlibrc for better Unicode support
        plt.rcParams['font.family'] = self.best_font['family']

        # Create a FontProperties object for later use
        self.hindi_font = FontProperties(family=self.best_font['family'], size=10)

        print(f"Using font: {self.best_font['family']} for Unicode text rendering")

    def get_unicode_font(self):
        # Default fallback
        default_font = {'family': 'DejaVu Sans', 'weight': 'normal'}

        # Platform-specific font recommendations
        system = platform.system()
        if system == 'Windows':
            fonts_to_try = ['Arial Unicode MS', 'Nirmala UI', 'Mangal', 'Arial']
        elif system == 'Darwin':
            fonts_to_try = ['Apple Color Emoji', 'Arial Unicode MS', 'Noto Sans']
        else:
            fonts_to_try = ['Noto Sans', 'Noto Color Emoji', 'DejaVu Sans', 'FreeSans']

        # Check which fonts are available
        available_fonts = [f.name for f in mpl.font_manager.fontManager.ttflist]

        # Find the first font that exists in our system
        for font in fonts_to_try:
            if any(font.lower() in af.lower() for af in available_fonts):
                return {'family': font, 'weight': 'normal'}

        # Return default font if none of the preferred fonts are available
        return default_font

    def truncate_text_properly(self, text, max_length=40):
        if len(text) <= max_length:
            return text

        char_count = 0
        truncated_text = ""

        # Process one character at a time to ensure emoji and Devanagari aren't cut
        for char in text:
            truncated_text += char
            char_count += 1

            if char_count >= max_length - 1:
                break

        # Add ellipsis to indicate truncation
        return truncated_text + "…"

    def contains_hindi(self, text):
        devanagari_range = range(0x0900, 0x097F + 1)  # Devanagari Unicode range
        return any(ord(char) in devanagari_range for char in text)

    def detect_emojis(self, text):
        emoji_scores = []

        # Extract all emojis from the text
        emojis_in_text = [c for c in text if c in self.emoji_sentiment]

        # Add sentiment scores for each emoji found
        for e in emojis_in_text:
            emoji_scores.append(self.emoji_sentiment.get(e, 0))

        return emoji_scores

    def preprocess(self, text):
        # Remove URLs, mentions, and hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)

        # For mixed scripts, we need to handle tokenization carefully
        tokens = word_tokenize(text)

        return text, tokens

    def get_sentiwordnet_scores(self, tokens):
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
        # Preprocess text and get tokens
        processed_text, tokens = self.preprocess(text)

        # Initialize sentiment score
        lexicon_score = 0

        # Check for emojis and add their sentiment weight
        emoji_scores = self.detect_emojis(text)
        if emoji_scores:
            emoji_sentiment = sum(emoji_scores)
            lexicon_score += emoji_sentiment

        # Check for emoticons in original text
        for emoticon, score in self.emoticons.items():
            if emoticon in text:
                lexicon_score += score

        # Word-based sentiment analysis using our lexicon
        for word in tokens:
            word_lower = word.lower()
            if word_lower in self.sentiment_dict:
                lexicon_score += self.sentiment_dict[word_lower]

        vader_scores = self.vader.polarity_scores(processed_text)
        vader_compound = vader_scores['compound']

        # Calculate average score from lexicon and VADER
        if len(tokens) > 0:
            # Adjust divisor based on emoji presence - emojis have stronger impact
            emoji_count = len(emoji_scores)
            word_count = len(tokens)

            # Emojis are weighted more heavily in shorter texts
            divisor = max(1, min((word_count - emoji_count * 0.5) / 3, 5))
            lexicon_score = lexicon_score / divisor

        # Combine lexicon score with VADER score, giving more weight to lexicon if emojis present
        if emoji_scores:
            base_score = (lexicon_score * 0.8) + (vader_compound * 0.2)
        else:
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
            'strength': abs(final_score),
            'emojis_found': bool(emoji_scores),
            'emoji_count': len(emoji_scores)
        }

    def analyze_multiple(self, sentences):
        results = []
        for sentence in sentences:
            sentiment_result = self.analyze_sentiment(sentence)
            results.append({
                'text': sentence,
                'sentiment_score': sentiment_result['score'],
                'sentiment_category': sentiment_result['category'],
                'sentiment_strength': sentiment_result['strength'],
                'emojis_found': sentiment_result['emojis_found'],
                'emoji_count': sentiment_result['emoji_count']
            })

        return results

    def plot_sentiment_histogram(self, results, save_path=None):
        # Extract data from results
        sentiment_scores = [r['sentiment_score'] for r in results]
        categories = [r['sentiment_category'] for r in results]
        emoji_counts = [r['emoji_count'] for r in results]

        # Enhanced text processing to better preserve emojis during truncation
        processed_texts = []
        for r in results:
            text = r['text']
            if len(text) > 40:
                emoji_positions = []
                for i, char in enumerate(text):
                    if char in emoji.EMOJI_DATA:
                        emoji_positions.append((i, char))

                if emoji_positions:
                    # Basic truncation first
                    trunc_text = self.truncate_text_properly(text[:35], max_length=35)

                    # Append emojis at the end for visibility if they would be cut off
                    if any(pos > 35 for pos, _ in emoji_positions):
                        emoji_chars = ' ' + ''.join([e[1] for e in emoji_positions if e[0] > 35])
                        processed_texts.append(trunc_text + "…" + emoji_chars)
                    else:
                        processed_texts.append(trunc_text)
                else:
                    # No emojis, use standard truncation
                    processed_texts.append(self.truncate_text_properly(text, max_length=40))
            else:
                processed_texts.append(text)

        contains_hindi_text = any(self.contains_hindi(text) for text in processed_texts)
        row_height = 0.6 if contains_hindi_text else 0.5
        fig_height = max(6, len(processed_texts) * row_height)

        fig, ax = plt.subplots(figsize=(12, fig_height), dpi=300)

        colors = {
            'Positive': '#4CAF50',
            'Negative': '#F44336',
            'Neutral': '#9E9E9E'
        }
        bar_colors = [colors[category] for category in categories]
        bars = ax.barh(range(len(processed_texts)), sentiment_scores, height=0.7, color=bar_colors)

        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)

        ax.set_yticks(range(len(processed_texts)))

        try:
            system = platform.system()
            if system == 'Darwin':
                emoji_hindi_font = self.hindi_font.copy()
                emoji_hindi_font.set_family(['Apple Color Emoji', *self.hindi_font.get_family()])
            elif system == 'Windows':
                emoji_hindi_font = self.hindi_font.copy()
                emoji_hindi_font.set_family(['Segoe UI Emoji', *self.hindi_font.get_family()])
            else:
                emoji_hindi_font = self.hindi_font.copy()
                emoji_hindi_font.set_family(['Noto Color Emoji', *self.hindi_font.get_family()])

            labels = ax.set_yticklabels(processed_texts, fontproperties=emoji_hindi_font)
        except:
            labels = ax.set_yticklabels(processed_texts, fontproperties=self.hindi_font)

        # Adjust font size for Hindi text
        if contains_hindi_text:
            for label in labels:
                label.set_fontsize(9)

        # Set x-axis label and title
        ax.set_xlabel('Sentiment Score: Negative (-1) to Neutral (0) to Positive (+1)', fontsize=12)
        ax.set_title('Hinglish-Devanagari Sentiment Analysis with Emoji Support',
                     fontsize=14, fontproperties=self.hindi_font)
        ax.set_xlim(-1.1, 1.1)
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)

        for i, bar in enumerate(bars):
            score = sentiment_scores[i]

            # Position logic
            if score < 0:
                label_position = score + 0.05
                alignment = 'left'
                txt_color = 'white' if score < -0.3 else 'black'
            else:
                label_position = score - 0.05
                alignment = 'right'
                txt_color = 'white' if score > 0.3 else 'black'

            # Add emoji indicator
            emoji_indicator = f" 🔍{emoji_counts[i]}" if emoji_counts[i] > 0 else ""

            ax.text(
                label_position,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.2f} ({categories[i]}){emoji_indicator}",
                ha=alignment,
                va='center',
                color=txt_color,
                fontweight='bold',
                fontsize=9,
                fontproperties=self.hindi_font
            )

        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color=colors['Positive']),
            plt.Rectangle((0, 0), 1, 1, color=colors['Neutral']),
            plt.Rectangle((0, 0), 1, 1, color=colors['Negative']),
            plt.Text(0, 0, "🔍", fontsize=10)
        ]
        ax.legend(legend_handles, ['Positive', 'Neutral', 'Negative', 'Emojis Found'],
                  loc='lower right', frameon=True, prop=self.hindi_font)
        plt.tight_layout()
        fig.subplots_adjust(left=0.2)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        return fig, ax


def read_mixed_sentences(devanagari_file):
    mixed_sentences = []

    if not os.path.exists(devanagari_file):
        print(f"Error: Mixed text file {devanagari_file} not found!")
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
    with open(input_file, 'r', encoding='utf-8') as file:
        sentences = [line.strip() for line in file if line.strip()]

    return sentences


def write_results_to_file(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("Hinglish-Devanagari Sentiment Analysis Results with Emoji Support\n")
        file.write("==========================================================\n\n")

        for i, result in enumerate(results):
            file.write(f"{i + 1}. Text: {result['text']}\n")
            file.write(f"   Sentiment Score: {result['sentiment_score']:.2f}\n")
            file.write(f"   Category: {result['sentiment_category']} (Strength: {result['sentiment_strength']:.2f})\n")

            # Add emoji information if present
            if result['emojis_found']:
                file.write(f"   Emojis found: Yes (Count: {result['emoji_count']})\n")

            file.write("\n")


def save_results_as_json(results, json_file):
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


def process_dataset(input_dir, output_dir, dataset_name):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define file paths
    devanagari_file = os.path.join(input_dir, "devanagari_output.txt")
    output_text_file = os.path.join(output_dir, f"sentiment_results_{dataset_name.lower()}.txt")
    json_output = os.path.join(output_dir, f"sentiment_results_{dataset_name.lower()}.json")
    histogram_file = os.path.join(output_dir, f"sentiment_histogram_{dataset_name.lower()}.png")

    # Create analyzer instance
    analyzer = HinglishSentimentAnalyzer()

    # Read mixed English-Devanagari sentences
    print(f"Reading mixed English-Devanagari sentences from {dataset_name} dataset...")
    sentences = read_mixed_sentences(devanagari_file)

    if not sentences:
        print(f"No sentences found in {devanagari_file}!")
        return

    print(f"Found {len(sentences)} sentences to analyze in {dataset_name} dataset.")

    # Analyze sentences
    print(f"Analyzing {dataset_name} mixed sentences (with emoji support)...")
    results = analyzer.analyze_multiple(sentences)

    # Write results to output file
    write_results_to_file(results, output_text_file)
    print(f"{dataset_name} analysis results written to {output_text_file}")

    # Save results as JSON for potential further processing
    save_results_as_json(results, json_output)
    print(f"{dataset_name} results also saved as JSON at {json_output}")

    # Create histogram visualization
    print(f"Generating {dataset_name} sentiment histogram...")
    analyzer.plot_sentiment_histogram(results, save_path=histogram_file)

    print(f"{dataset_name} analysis complete! Histogram saved to: {histogram_file}")

    # Display summary of results
    categories = [r['sentiment_category'] for r in results]
    pos_count = categories.count('Positive')
    neg_count = categories.count('Negative')
    neu_count = categories.count('Neutral')

    emoji_texts = [r for r in results if r['emojis_found']]

    print(f"\nSummary of {dataset_name} Results:")
    print(f"Total sentences analyzed: {len(results)}")
    print(f"Positive: {pos_count} ({pos_count / len(results) * 100:.1f}%)")
    print(f"Negative: {neg_count} ({neg_count / len(results) * 100:.1f}%)")
    print(f"Neutral: {neu_count} ({neu_count / len(results) * 100:.1f}%)")
    print(f"Sentences with emojis: {len(emoji_texts)} ({len(emoji_texts) / len(results) * 100:.1f}%)")

    return results


def main():
    # Define directories
    normalized_dir_unofficial = "Normalized output/Unofficial"
    normalized_dir_official = "Normalized output/Official"

    sentiment_dir_unofficial = "Sentiment Analysis/Unofficial"
    sentiment_dir_official = "Sentiment Analysis/Official"

    # Process unofficial dataset
    print("\n===== Processing Unofficial Dataset =====")
    unofficial_results = process_dataset(
        normalized_dir_unofficial,
        sentiment_dir_unofficial,
        "Unofficial"
    )

    # Process official dataset
    print("\n===== Processing Official Dataset =====")
    official_results = process_dataset(
        normalized_dir_official,
        sentiment_dir_official,
        "Official"
    )

    # Create combined output directory
    combined_dir = "Sentiment Analysis/Combined"
    os.makedirs(combined_dir, exist_ok=True)

    # Create combined visualization if both datasets were processed successfully
    if unofficial_results and official_results:
        print("\n===== Creating Combined Analysis =====")

        # Merge results
        all_results = []
        for result in unofficial_results:
            result['dataset'] = 'Unofficial'
            all_results.append(result)
        for result in official_results:
            result['dataset'] = 'Official'
            all_results.append(result)

        # Save combined results
        combined_json = os.path.join(combined_dir, "sentiment_results_combined.json")
        save_results_as_json(all_results, combined_json)

        # Create combined summary file
        combined_text = os.path.join(combined_dir, "sentiment_results_combined.txt")
        with open(combined_text, 'w', encoding='utf-8') as file:
            file.write("Combined Hinglish-Devanagari Sentiment Analysis Results\n")
            file.write("===========================================\n\n")

            file.write("Unofficial Dataset Summary:\n")
            file.write(f"- Total sentences: {len(unofficial_results)}\n")
            file.write(f"- Positive: {sum(1 for r in unofficial_results if r['sentiment_category'] == 'Positive')}\n")
            file.write(f"- Negative: {sum(1 for r in unofficial_results if r['sentiment_category'] == 'Negative')}\n")
            file.write(f"- Neutral: {sum(1 for r in unofficial_results if r['sentiment_category'] == 'Neutral')}\n\n")

            file.write("Official Dataset Summary:\n")
            file.write(f"- Total sentences: {len(official_results)}\n")
            file.write(f"- Positive: {sum(1 for r in official_results if r['sentiment_category'] == 'Positive')}\n")
            file.write(f"- Negative: {sum(1 for r in official_results if r['sentiment_category'] == 'Negative')}\n")
            file.write(f"- Neutral: {sum(1 for r in official_results if r['sentiment_category'] == 'Neutral')}\n\n")

            file.write("Combined Analysis:\n")
            file.write(f"- Total sentences: {len(all_results)}\n")
            file.write(f"- Positive: {sum(1 for r in all_results if r['sentiment_category'] == 'Positive')}\n")
            file.write(f"- Negative: {sum(1 for r in all_results if r['sentiment_category'] == 'Negative')}\n")
            file.write(f"- Neutral: {sum(1 for r in all_results if r['sentiment_category'] == 'Neutral')}\n")

        print(f"Combined analysis saved to {combined_dir}")

    print("\nAll sentiment analysis processes completed!")


if __name__ == "__main__":
    main()