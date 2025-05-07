import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.metrics.distance import edit_distance
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle
import re
import os
import nltk
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')


class HinglishWordClassifier:
    def __init__(self):
        self.vectorizer = None
        self.model = None

        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

        # List of words that should not be lemmatized
        self.do_not_lemmatize = {
            "as", "was", "has", "is", "this", "his", "hers", "its", "theirs",
            "yours", "bus", "pass", "class", "glass", "grass", "brass", "us",
            "yes", "kiss", "miss", "mess", "less", "toss", "boss", "loss"
        }

        # Initialize normalization dictionary for English abbreviations and contractions
        self.english_norm_dict = {
            "pls": "please", "plz": "please", "u": "you", "r": "are",
            "ur": "your", "thx": "thanks", "wud": "would", "wht": "what",
            "abt": "about", "bcoz": "because", "cuz": "because", "b4": "before",
            "gr8": "great", "btw": "by the way", "ty": "thank you",
            "gn": "good night", "gm": "good morning", "idk": "i don't know",
            "lol": "laugh out loud", "omg": "oh my god", "dm": "direct message",
            "tbh": "to be honest", "lmk": "let me know", "brb": "be right back",
            "rn": "right now", "irl": "in real life", "fb": "facebook",
            "cya": "see you", "asap": "as soon as possible", "thnx": "thanks",
            "msg": "message", "tmrw": "tomorrow", "tdy": "today",
            "tho": "though", "thru": "through", "gonna": "going to",
            "wanna": "want to", "gotta": "got to", "dunno": "don't know",
            "yep": "yes", "nope": "no", "wassup": "what's up",
            "info": "information", "pic": "picture", "pics": "pictures",
            "convo": "conversation", "coz": "because", "bcz": "because",
            "2day": "today", "2moro": "tomorrow", "4get": "forget",
            "c u": "see you", "tel": "tell", "tol": "told"
        }

        # Dictionary for Hindi-specific normalization
        self.hindi_norm_dict = {
            "nhi": "nahi", "kese": "kaise", "acha": "achha", "thik": "theek",
            "pyar": "pyaar", "kuch": "kuchh", "kam": "kaam", "jada": "zyada",
            "kaha": "kahaan", "me": "main", "mai": "main", "h": "hai",
            "k": "ke", "n": "ne", "mtlb": "matlab", "accha": "achha",
            "hyn": "haan", "haa": "haan", "hn": "haan", "mje": "mujhe",
            "yr": "yaar", "ha": "haan", "hm": "hum", "shyd": "shayad",
            "abi": "abhi", "abhi": "abhi", "k liye": "ke liye",
            "krna": "karna", "kro": "karo", "ache": "achhe",
            "aaj kl": "aaj kal", "aajkl": "aaj kal", "koi ni": "koi nahi",
            "koi nh": "koi nahi", "sb": "sab", "agr": "agar", "vo": "woh",
            "krte": "karte", "ho gya": "ho gaya", "hogya": "ho gaya",
            "kya hua": "kya hua", "kyun": "kyon", "ghr": "ghar",
            "fr": "fir", "zyda": "zyada", "kha": "kahan", "kb": "kab",
            "ke lie": "ke liye", "fikar": "fikr", "bhaut": "bahut",
            "bht": "bahut", "bht": "bahut", "bhot": "bahut"
        }

    def load_data(self, file_path):
        """Load data from a file with word-label pairs."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                parts = line.split('\t')
                if len(parts) == 2:
                    word, lang = parts
                    data.append((word, lang))

        return pd.DataFrame(data, columns=['word', 'language'])

    def extract_features(self, X, fit=False):
        """Extract character n-gram features from words."""
        if fit:
            self.vectorizer = TfidfVectorizer(
                analyzer='char',
                ngram_range=(1, 4),  # Use character 1-4 grams
                min_df=2,  # Minimum document frequency
                max_features=3000  # Limit features to prevent overfitting
            )
            return self.vectorizer.fit_transform(X)
        else:
            return self.vectorizer.transform(X)

    def train(self, data_file, test_size=0.2):
        """Train the model using the data from the specified file."""
        # Load data
        df = self.load_data(data_file)

        # Split into features and target
        X = df['word'].values
        y = df['language'].values

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Extract features
        X_train_features = self.extract_features(X_train, fit=True)
        X_test_features = self.extract_features(X_test)

        # Train the model
        self.model = LogisticRegression(max_iter=1000, C=10.0)
        self.model.fit(X_train_features, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_features)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy:.4f}")
        cm = confusion_matrix(y_test, y_pred, labels=["EN", "HI"])
        labels = ["EN", "HI"]
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix - Training Data")
        plt.tight_layout()
        plt.savefig("training_data_matrix.png")
        plt.close()
        return accuracy

    def save_model(self, file_path="hinglish_model.pkl"):
        """Save the trained model and vectorizer."""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained yet. Call train() first.")

        with open(file_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer
            }, f)

        print(f"Model saved to {file_path}")

    def load_model(self, file_path="hinglish_model.pkl"):
        """Load a pre-trained model and vectorizer."""
        with open(file_path, 'rb') as f:
            saved_data = pickle.load(f)

        self.model = saved_data['model']
        self.vectorizer = saved_data['vectorizer']
        print(f"Model loaded from {file_path}")

    def normalize_word(self, word):
        """
        Normalize a single word using NLTK and custom rules.
        Handle English and Hindi words differently.
        """
        # Remove punctuation for normalization, but keep track of it
        word_clean = re.sub(r'[^\w\s]', '', word.lower())
        punctuation = ''.join(c for c in word if c in string.punctuation)
        punct_positions = []
        if word.startswith(punctuation):
            punct_positions.append('start')
        if word.endswith(punctuation):
            punct_positions.append('end')

        # Try to determine if word is likely English or Hindi
        # (This is a simple heuristic, could be improved)
        is_likely_english = bool(re.search(r'[a-zA-Z]', word_clean))

        # Normalize based on language
        if is_likely_english:
            # For English words, use custom replacements and NLTK
            normalized = word_clean
            # Apply common text replacements
            normalized = self.english_norm_dict.get(normalized, normalized)

            # Only lemmatize if it's not in our do_not_lemmatize list
            if normalized not in self.do_not_lemmatize:
                try:
                    # Check if lemmatization would change the word
                    lemmatized = self.lemmatizer.lemmatize(normalized)
                    # Only use lemmatized form if it doesn't drastically change the word
                    if len(lemmatized) > 1:  # Avoid single-letter lemmatizations
                        normalized = lemmatized
                except:
                    pass  # Skip if lemmatization fails
        else:
            # For Hindi/Hinglish words, use our custom replacements
            normalized = self.hindi_norm_dict.get(word_clean, word_clean)

        # Reattach punctuation if it existed
        if 'start' in punct_positions:
            normalized = punctuation + normalized
        if 'end' in punct_positions:
            normalized = normalized + punctuation

        return normalized

    def normalize_text(self, text):
        """
        Apply comprehensive text normalization.
        """
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text).strip()

        # Correct multiple punctuation
        text = re.sub(r'([!?])\1+', r'\1', text)  # Convert !! or ??? to single ! or ?
        text = re.sub(r'\.{2,}', '...', text)  # Normalize ellipses to three dots

        # Handle contractions
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"'d", " would", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'t", " not", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'m", " am", text)

        # Tokenize
        tokens = word_tokenize(text)

        # Apply word-level normalization
        normalized_tokens = [self.normalize_word(token) for token in tokens]

        # Join tokens back together
        normalized_text = ' '.join(normalized_tokens)

        return normalized_text

    def tokenize(self, text):
        """Tokenize text into words using NLTK."""
        return word_tokenize(text.lower())

    def process_file(self, input_file, normalized_file='normalized_hinglish.txt',
                     tagged_file='hinglish_tagged.txt', metrics_file='normalization_metrics.txt'):
        """
        Process input file in three steps:
        1. Normalize text and save to normalized_file
        2. Classify normalized text and save to tagged_file
        3. Calculate metrics between original and normalized text
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Read input sentences
        original_sentences = []
        with open(input_file, 'r', encoding='utf-8') as in_file:
            for line in in_file:
                line = line.strip()
                if line:  # Skip empty lines
                    original_sentences.append(line)

        # Step 1: Normalize sentences
        normalized_sentences = []
        for sentence in original_sentences:
            # Use our custom normalization which handles Hinglish better
            normalized = self.normalize_text(sentence)
            normalized_sentences.append(normalized)

        # Save normalized text
        with open(normalized_file, 'w', encoding='utf-8') as norm_file:
            for i, (original, normalized) in enumerate(zip(original_sentences, normalized_sentences)):
                norm_file.write(f"Original [{i + 1}]: {original}\n")
                norm_file.write(f"Normalized [{i + 1}]: {normalized}\n\n")

        # Step 2: Classify and tag normalized text
        with open(tagged_file, 'w', encoding='utf-8') as tag_file:
            for i, normalized in enumerate(normalized_sentences):
                tag_file.write(f"Sentence [{i + 1}]: {normalized}\n")

                # Split into words and classify
                words = self.tokenize(normalized)
                if words:
                    predictions = self.predict(words)

                    # Write word-tag pairs
                    tag_file.write("Word\tLanguage\n")
                    for word, lang in predictions:
                        tag_file.write(f"{word}\t{lang}\n")
                tag_file.write("\n")

        # Step 3: Calculate metrics between original and normalized
        total_bleu_scores = []
        total_edit_distances = []
        smooth_fn = SmoothingFunction().method1

        for original, normalized in zip(original_sentences, normalized_sentences):
            orig_tokens = self.tokenize(original)
            norm_tokens = self.tokenize(normalized)

            if orig_tokens and norm_tokens:
                bleu_score = sentence_bleu([orig_tokens], norm_tokens, smoothing_function=smooth_fn)
                total_bleu_scores.append(bleu_score)

            edit_dist = edit_distance(original, normalized)
            total_edit_distances.append(edit_dist)

        # Calculate average metrics
        avg_bleu = np.mean(total_bleu_scores) if total_bleu_scores else 0.0
        avg_edit_distance = np.mean(total_edit_distances) if total_edit_distances else 0.0

        # Write metrics
        with open(metrics_file, 'w', encoding='utf-8') as metrics:
            metrics.write("### Normalization Metrics ###\n")
            metrics.write(f"Average BLEU Score: {avg_bleu:.4f}\n")
            metrics.write(f"Average Edit Distance: {avg_edit_distance:.4f}\n\n")

            metrics.write("Individual Sentence Metrics:\n")
            for i, (original, normalized) in enumerate(zip(original_sentences, normalized_sentences)):
                metrics.write(f"Sentence {i + 1}:\n")
                metrics.write(f"  Original: {original}\n")
                metrics.write(f"  Normalized: {normalized}\n")

                if i < len(total_bleu_scores):
                    metrics.write(f"  BLEU Score: {total_bleu_scores[i]:.4f}\n")
                else:
                    metrics.write("  BLEU Score: N/A\n")

                metrics.write(f"  Edit Distance: {total_edit_distances[i]}\n\n")

        print(f"\n--- Results from Processing {input_file} ---")
        print(f"Processed {len(original_sentences)} sentences")
        print(f"Average BLEU Score: {avg_bleu:.4f}")
        print(f"Average Edit Distance: {avg_edit_distance:.4f}")

        # Display a few examples
        print("\nSample Results:")
        for i in range(min(3, len(original_sentences))):
            print(f"Original: {original_sentences[i]}")
            print(f"Normalized: {normalized_sentences[i]}")
            if i < len(total_bleu_scores):
                print(f"BLEU Score: {total_bleu_scores[i]:.4f}")
            print(f"Edit Distance: {total_edit_distances[i]}\n")

        print(f"Normalized text saved to: {normalized_file}")
        print(f"Tagged text saved to: {tagged_file}")
        print(f"Metrics saved to: {metrics_file}")

        return normalized_sentences, avg_bleu, avg_edit_distance

    def predict(self, words):
        """Predict language for a list of words."""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained yet. Call train() first or load a model.")

        # Convert input to list if it's a single word
        if isinstance(words, str):
            words = [words]

        # Extract features
        features = self.extract_features(words)

        # Predict
        predictions = self.model.predict(features)

        # Return predictions
        return list(zip(words, predictions))


# Main function
def main():
    # Initialize the classifier
    classifier = HinglishWordClassifier()

    # Train the model using train.txt
    print("Training model from train.txt...")
    classifier.train(data_file="train.txt")

    # Save the model
    classifier.save_model()

    # Process input file
    input_file = "hinglish_sentences.txt"
    print(f"Processing {input_file}...")
    classifier.process_file(
        input_file=input_file,
        normalized_file="normalized_hinglish.txt",
        tagged_file="hinglish_tagged.txt",
        metrics_file="normalization_metrics.txt"
    )


if __name__ == "__main__":
    main()