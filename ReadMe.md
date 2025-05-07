# Hinglish Normalization and Sentiment Analysis

This project implements a system for normalization and sentiment analysis of Hinglish text (Hindi–English code-mixed language). The system consists of two main components:

1. `hinglish_norm.py`: Normalizes Hinglish text and classifies words as Hindi or English  
2. `sentiment.py`: Performs sentiment analysis on the normalized Hinglish text  

## Requirements

### Required Packages

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk textblob emoji install unicodedata
```

### NLTK Downloads

The system will automatically attempt to download necessary NLTK resources, but you can pre-download them:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')
```

## Workflow

### Step 1: Text Normalization

If you want to add more lines use the cleaned_chat_unofficial or official to
add the necessary text by just copy and pasting into the hinglish_sentences_unofficial or
official respectively for a wider amount of analysis, currently only 100 lines are in each file.

Run the `hinglish_norm.py` script first to normalize the Hinglish text:

```bash
python hinglish_norm.py
```

This script:
- Normalizes Hinglish text (corrects spellings, standardizes abbreviations, etc.)  
- Classifies words as Hindi (HI), English (EN), or Emoji (EM)  
- Transliterates Hindi words from Latin script to Devanagari script  
- Creates evaluation metrics for normalization quality  

#### Outputs

The script creates the following directories and files:
- `Normalized output/Unofficial/`  
  - `normalized_hinglish.txt`: Normalized text in Latin script  
  - `hinglish_tagged.txt`: Words tagged with language labels (EN/HI/EM)  
  - `devanagari_output.txt`: Mixed text with Hindi words in Devanagari script  
  - `normalization_metrics.txt`: BLEU scores and edit distances  
- `Normalized output/Official/` (same files as above for the official dataset)  
- `models/hinglish_model.pkl`: Trained classifier model  
- `training_data_matrix.png`: Confusion matrix visualization  
- `training_log.txt`: F1, recall and precision score along with total examples used in training 
our normalizer

### Step 2: Sentiment Analysis

After normalization, run the sentiment analysis script:

```bash
python sentiment.py
```

This script:
- Reads the normalized mixed English–Devanagari text from the previous step  
- Analyzes sentiment using a combination of techniques  
- Handles emojis and emoticons as part of sentiment analysis  
- Creates visualizations of sentiment scores through a histogram with positive and negative scores representing the positive and negative emotions respectively 

#### Outputs

The script creates the following directories and files:
- `Sentiment Analysis/Unofficial/`  
  - `sentiment_results_unofficial.txt`: Sentiment analysis results in text format  
  - `sentiment_results_unofficial.json`: Results in JSON format for further processing  
  - `sentiment_histogram_unofficial.png`: Visualization of sentiment scores  
- `Sentiment Analysis/Official/` (same files as above for the official dataset)  
- `Sentiment Analysis/Combined/`  
  - `sentiment_results_combined.txt`: Summary of both datasets  
  - `sentiment_results_combined.json`: Combined results in JSON format  

## Example Use Case

This system can be used for:  
1. Normalizing code-mixed Hinglish social media text  
2. Classifying words in Hinglish text by language  
3. Converting Hindi words in Latin script to Devanagari  
4. Analyzing sentiment of Hinglish texts with emoji support  

The system handles both unofficial and official datasets and provides comprehensive metrics and visualizations.