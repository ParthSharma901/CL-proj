import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.metrics.distance import edit_distance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle
import re
import os


class HinglishWordClassifier:
    def __init__(self):
        self.vectorizer = None
        self.model = None

        # Initialize normalization dictionary for Hinglish
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
        """Normalize a single word using the dictionary or return the original."""
        # Remove punctuation for lookup, but keep original for return
        word_clean = re.sub(r'[^\w\s]', '', word.lower())

        # Get normalized form or original if not in dictionary
        normalized = self.normalization_dict.get(word_clean, word_clean)

        # If word had punctuation, transfer it to the normalized form
        if word != word_clean:
            # Find the punctuation
            punct_match = re.search(r'[^\w\s]+', word)
            if punct_match:
                punct = punct_match.group(0)
                # Check if punctuation is at beginning, end, or both
                if word.startswith(punct):
                    normalized = punct + normalized
                if word.endswith(punct):
                    normalized = normalized + punct

        return normalized

    def normalize_sentence(self, sentence):
        """
        Normalize a Hinglish sentence by:
        1. Converting to lowercase
        2. Fixing spelling mistakes and short forms
        3. Removing extra spaces
        """
        # Convert to lowercase
        sentence = sentence.lower()

        # Split into words while preserving spaces 
        tokens = re.findall(r'\S+|\s+', sentence)

        # Normalize each word
        normalized_tokens = []
        for token in tokens:
            if token.strip():  # If it's a word
                normalized_tokens.append(self.normalize_word(token))
            else:  # If it's whitespace
                normalized_tokens.append(token)

        # Combine and clean up extra spaces
        normalized = ''.join(normalized_tokens)
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        return normalized

    def tokenize(self, text):
        """Tokenize Hinglish text into words."""
        return re.findall(r'\b\w+\b', text.lower())

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
            normalized = self.normalize_sentence(sentence)
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