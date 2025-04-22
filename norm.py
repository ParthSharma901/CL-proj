import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import re
import os
from datetime import datetime


class HinglishWordClassifier:
    def __init__(self):
        self.vectorizer = None
        self.model = None

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

    def train(self, data_file, test_size=0.2, log_file='training_log.txt'):
        """Train the model using the data from the specified file and log results."""
        # Load data
        df = self.load_data(data_file)

        # Create log file
        with open(log_file, 'w', encoding='utf-8') as log:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log.write(f"Training started at: {timestamp}\n")
            log.write(f"Training data file: {data_file}\n")
            log.write(f"Number of examples: {len(df)}\n\n")

            # Split into features and target
            X = df['word'].values
            y = df['language'].values

            # Count classes
            classes, counts = np.unique(y, return_counts=True)
            for cls, count in zip(classes, counts):
                log.write(f"Class {cls}: {count} examples\n")
            log.write("\n")

            # Split into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            log.write(f"Training set: {len(X_train)} examples\n")
            log.write(f"Test set: {len(X_test)} examples\n\n")

            # Extract features
            log.write("Extracting features...\n")
            X_train_features = self.extract_features(X_train, fit=True)
            X_test_features = self.extract_features(X_test)

            # Train the model
            log.write("Training the model...\n")
            self.model = LogisticRegression(max_iter=1000, C=10.0)
            self.model.fit(X_train_features, y_train)

            # Evaluate
            y_pred = self.model.predict(X_test_features)

            # Log metrics
            accuracy = accuracy_score(y_test, y_pred)
            log.write(f"Model accuracy: {accuracy:.4f}\n\n")
            log.write("Classification Report:\n")
            log.write(classification_report(y_test, y_pred))
            log.write("\n")

            # Log some example predictions
            log.write("Example predictions:\n")
            for word, true_label, pred_label in zip(X_test[:10], y_test[:10], y_pred[:10]):
                log.write(f"Word: '{word}', True: {true_label}, Predicted: {pred_label}\n")

            # Log completion
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log.write(f"\nTraining completed at: {timestamp}\n")

        print(f"Model trained with accuracy: {accuracy:.4f}")
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

    def classify_file(self, input_file, output_file):
        """
        Classify each line in the input file and write results to output file.
        Each line should contain a word or sentence.
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        with open(input_file, 'r', encoding='utf-8') as in_file, \
                open(output_file, 'w', encoding='utf-8') as out_file:

            for line in in_file:
                line = line.strip()
                if not line:
                    out_file.write('\n')
                    continue

                # Split line into words
                words = re.findall(r'\b\w+\b', line)

                # Get predictions
                predictions = self.predict(words)

                # Write results to output file
                for word, lang in predictions:
                    out_file.write(f"{word}\t{lang}\n")

                # Add an empty line between sentences
                out_file.write('\n')

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


# Main function with fixed filenames as requested
def main():
    # Initialize the classifier
    classifier = HinglishWordClassifier()

    # Train the model using train.txt
    print("Training model from train.txt...")
    classifier.train(data_file="train.txt")

    # Save the model
    classifier.save_model()

    # Process input.txt and generate output.txt
    print("Processing input.txt...")
    classifier.classify_file(input_file="hinglish_sentences.txt", output_file="normalized_hinglish.txt")
    print("Results saved to output.txt")


if __name__ == "__main__":
    main()