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
        self.do_not_lemmatize = {"as", "was", "has", "is", "this", "his", "hers", "its", "theirs", "yours", "bus",
                                 "pass", "class", "glass", "grass", "brass", "us", "yes", "kiss", "miss", "mess",
                                 "less", "toss", "boss", "loss"}

        # Initialize normalization dictionary for English abbreviations and contractions
        self.english_norm_dict = {
            # Original abbreviations
            "pls": "please", "plz": "please", "u": "you", "r": "are", "ur": "your", "thx": "thanks",
            "wud": "would", "wht": "what", "abt": "about", "bcoz": "because", "cuz": "because", "b4": "before",
            "gr8": "great", "btw": "by the way", "ty": "thank you", "gn": "good night", "gm": "good morning",
            "idk": "i don't know", "lol": "laugh out loud", "omg": "oh my god", "dm": "direct message",
            "tbh": "to be honest", "lmk": "let me know", "brb": "be right back", "rn": "right now",
            "irl": "in real life", "fb": "facebook", "cya": "see you", "asap": "as soon as possible",
            "thnx": "thanks", "msg": "message", "tmrw": "tomorrow", "tdy": "today", "tho": "though",
            "thru": "through", "gonna": "going to", "wanna": "want to", "gotta": "got to", "dunno": "don't know",
            "yep": "yes", "nope": "no", "wassup": "what's up", "info": "information", "pic": "picture",
            "pics": "pictures", "convo": "conversation", "coz": "because", "bcz": "because", "2day": "today",
            "2moro": "tomorrow", "4get": "forget", "c u": "see you", "tel": "tell", "tol": "told","lmao":"laughing my ass off",

            "aaf": "always and forever", "aab": "average at best", "aak": "alive and kicking",
            "aamof": "as a matter of fact", "aamoi": "as a matter of interest", "aap": "always a pleasure",
            "aar": "at any rate", "aas": "alive and smiling", "abd": "already been done",
            "abh": "actual bodily harm", "abk": "always be knolling", "abt": "about",
            "acd": "alt control delete", "ace": "access control entry", "bak": "back at keyboard",
            "bau": "business as usual", "bb": "bye bye", "bb4n": "bye bye for now",
            "bbbg": "bye bye be good", "bbc": "bring beer and chips", "bbiab": "be back in a bit",
            "bbn": "bye bye now", "bbq": "barbecue", "bbr": "burnt beyond repair",
            "bbs": "be back soon", "bbsd": "be back soon darling", "bbsl": "be back sooner or later",
            "bbt": "be back tomorrow", "bcbs": "big company, big school", "bcnu": "be seeing you",
            "bd": "big deal", "bdn": "big damn number", "beos": "nudge", "bf": "boy friend",
            "bfd": "big frickin deal", "bff": "best friend forever", "cil": "check in later",
            "clm": "career limiting move", "cm": "call me", "cmb": "call me back",
            "cmf": "count my fingers", "cmiw": "correct me if i'm wrong", "cmu": "crack me up",
            "cob": "close of business", "col": "chuckle out loud", "coo": "cool",
            "cot": "circle of trust", "cpg": "consumer packaged goods", "crb": "come right back",
            "csa": "cool sweet awesome", "cto": "check this out", "dba": "doing business as",
            "dbd": "don't be dumb", "ddg": "drop dead gorgeous", "df": "dear friend",
            "dfik": "darn if i know", "dftba": "don't forget to be awesome", "dfwly": "don't forget who loves you",
            "dga": "don't go anywhere", "dgt": "don't go there", "dhyb": "don't hold your breath",
            "diku": "do i know you", "diy": "do it yourself", "djm": "don't judge me",
            "dk": "don't know", "dkdc": "don't know don't care", "dmi": "don't mention it",
            "dnbl8": "do not be late", "dnc": "does not compute", "doa": "dead on arrival",
            "doc": "drug of choice", "doe": "depends on experience", "doei": "goodbye",

            # Additional abbreviations
            "sem": "semester", "ts": "this shit", "pmo": "pissing me off",

            # Slang words from ANNEXURE B
            "2mrw": "tomorrow", "h/o": "hold on", "2d4": "to die for", "h/p": "hold please",
            "2day": "today", "h2cus": "hope to see you soon", "2dloo": "toodle oo", "h2s": "here to stay",
            "2g2b4g": "too good to be forgotten", "h4u": "hot for you", "2g2bt": "too good to be true",
            "h4xx0r": "hacker", "2g4u": "too good for you", "h8": "hate", "cul": "cool",
            "j/c": "just checking", "d2d": "day-to-day", "j/j": "just joking", "dt": "date",
            "j/k": "just kidding", "da": "there", "j/p": "just playing", "every1": "everyone",
            "j/w": "just wondering", "evre1": "every one", "j2lyk": "just to let you know",
            "e2eg": "each to his/her own", "j4f": "just for fun", "g2g": "got to go",
            "j4g": "just for grins", "g2glys": "got to go love ya so", "jft": "just for today",
            "g4i": "go for it", "l@u": "laughing at you", "g4n": "good for nothing", "l8": "late",
            "g98t": "good night", "lng": "long", "g9": "genius", "lst": "last", "g8": "great",
            "ly4e": "love you forever", "gud": "good", "m/f": "male or female", "gd": "good",
            "m2ny": "me too, not yet", "h&k": "hug and kiss", "m4c": "meet for coffee", "2l8": "too late",
            "nc": "nice", "2u2": "to you too", "no1": "no one", "2mor": "tomorrow", "ntng": "nothing",
            "2qt": "too cute", "nw": "new", "2n8": "tonight", "nxt": "next", "3sum": "threesome",
            "o3": "out of office", "4col": "for crying out loud", "sm1": "someone", "4e": "forever",
            "w4u": "waiting for you", "4eae": "forever and ever", "w8": "wait", "4f?": "for friends?",
            "w8 4 me": "wait for me", "4nr48": "for nothing", "w8n": "waiting", "a2d": "agree to disagree",
            "w9": "wife in room", "a3": "anywhere, anytime, anyplace", "wan2": "want to",
            "abt": "about", "wan2tlk": "want to talk?", "abl": "able", "wht": "what",
            "abt2": "about to", "wid": "with", "awsm": "awesome", "spk": "speak",
            "any1": "anyone", "wrst": "worst", "b2a": "business-to-anyone"
        }

        # Dictionary for Hindi-specific normalization
        self.hindi_norm_dict = {
            "nhi": "nahi", "kese": "kaise", "acha": "achha", "thik": "theek", "pyar": "pyaar",
            "kuch": "kuchh", "kam": "kaam", "jada": "zyada", "kaha": "kahaan", "me": "main",
            "mai": "main", "h": "hai", "k": "ke", "n": "ne", "mtlb": "matlab", "accha": "achha",
            "hyn": "haan", "haa": "haan", "hn": "haan", "mje": "mujhe", "yr": "yaar", "ha": "haan",
            "hm": "hum", "shyd": "shayad", "abi": "abhi", "abhi": "abhi", "k liye": "ke liye",
            "krna": "karna", "kro": "karo", "ache": "achhe", "aaj kl": "aaj kal", "aajkl": "aaj kal",
            "koi ni": "koi nahi", "koi nh": "koi nahi", "sb": "sab", "agr": "agar", "vo": "woh",
            "krte": "karte", "ho gya": "ho gaya", "hogya": "ho gaya", "kya hua": "kya hua",
            "kyun": "kyon", "ghr": "ghar", "fr": "fir", "zyda": "zyada", "kha": "kahan", "kb": "kab",
            "ke lie": "ke liye", "fikar": "fikr", "bhaut": "bahut", "bht": "bahut", "bhot": "bahut"
        }

        # Set of known English words for correcting repeated characters
        # This will be populated in load_english_dictionary
        self.english_words = set()
        self.load_english_dictionary()

        # Initialize the Hindi to Devanagari transliteration mapping
        self.init_transliteration_mapping()

    def init_transliteration_mapping(self):
        """Initialize the Hindi to Devanagari transliteration mapping from ANNEXURE D and E."""
        # From ANNEXURE D (Barakhadi for Transliteration)
        self.hindi_to_dev_map = {
            "a": "अ", "aa": "आ", "i": "इ", "ee": "ई", "u": "उ", "oo": "ऊ", "e": "ए", "ea": "ए",
            "ai": "ऐ", "ei": "ऐ", "o": "ओ", "ou": "औ", "au": "औ", "an": "अं", "am": "अं",
            "ah": "अः", "aha": "अ:", "ru": "ऋ",
            "k": "क", "ka": "क", "kaa": "का", "ki": "िक", "kee": "की", "ku": "कु", "koo": "कू",
            "ke": "के", "kai": "कै", "ko": "को", "kau": "कौ", "kan": "कं", "kam": "कं", "kah": "कः",
            "kh": "ख", "kha": "ख", "khaa": "खा", "khi": "िख", "khee": "खी", "khu": "खु", "khoo": "खू",
            "khe": "खे", "khai": "खै", "kho": "खो", "khau": "खौ", "khan": "खं", "kham": "खं", "khah": "खः",
            "g": "ग", "ga": "गा", "gaa": "गा", "gi": "िग", "gee": "गी", "gu": "गु", "goo": "गू",
            "ge": "गे", "gai": "गै", "go": "गो", "gau": "गौ", "gan": "गं", "gam": "गं", "gah": "गः",
            "gh": "घ", "gha": "घा", "ghaa": "घा", "ghi": "िघ", "ghee": "घी", "ghu": "घु", "ghoo": "घू",
            "ghe": "घे", "ghai": "घै", "gho": "घो", "ghau": "घौ", "ghan": "घं", "gham": "घं", "ghaah": "घः",
            "ch": "च", "cha": "च", "chaa": "चा", "chi": "िच", "chee": "ची", "chu": "चु", "choo": "चू",
            "che": "चे", "chai": "चै", "cho": "चो", "chau": "चौ", "chan": "चं", "chah": "चः",
            "chha": "छ", "chhaa": "छा", "chhi": "िछ", "chhee": "छी", "chhu": "छु", "chhoo": "छू",
            "chhe": "छे", "chhai": "छै", "chho": "छो", "chhau": "छौ", "chhan": "छं", "chham": "छं",
            "j": "ज", "ja": "ज", "jaa": "जा", "ji": "िज", "jee": "जी", "ju": "जु", "joo": "जू",
            "je": "जे", "jai": "जै", "jo": "जो", "jau": "जौ", "jan": "जं", "jam": "जं", "jah": "जः",
            "z": "झ", "za": "झ", "zaa": "झा", "zi": "िझ", "zee": "झी", "zu": "झु", "zoo": "झू",
            "ze": "झे", "zai": "झै", "zo": "झो", "zau": "झौ", "zan": "झं", "zam": "झं", "zah": "झः",
            "t": "त", "ta": "ता", "taa": "ता", "ti": "ित", "tee": "ती", "tu": "तु", "too": "तू",
            "te": "ते", "tai": "तै", "to": "तो", "tau": "तौ", "tan": "तं", "tam": "तं", "tah": "तः",
            "th": "थ", "tha": "थ", "thaa": "था", "thi": "िथ", "thee": "थी", "thu": "थु", "thoo": "थू",
            "the": "थे", "thai": "थै", "tho": "थो", "thau": "थौ", "than": "थं", "tham": "थं", "thah": "थः",
            "d": "द", "da": "द", "daa": "दा", "di": "िद", "dee": "दी", "du": "दु", "doo": "दू",
            "de": "दे", "dai": "दै", "do": "दो", "dau": "दौ", "dan": "दं", "dam": "दं", "dah": "दः",
            "dh": "ध", "dha": "ध", "dhaa": "धा", "dhi": "िध", "dhee": "धी", "dhu": "धु", "dhoo": "धू",
            "dhe": "धे", "dhai": "धै", "dho": "धो", "dhau": "धौ", "dhan": "धं", "dham": "धं", "dhah": "धः",
            "n": "न", "na": "न", "naa": "ना", "ni": "िन", "nee": "नी", "nu": "नु", "noo": "नू",
            "ne": "ने", "nai": "नै", "no": "नो", "nau": "नौ", "nan": "नं", "nam": "नं", "nah": "नः",
            "p": "प", "pa": "प", "paa": "पा", "pi": "िप", "pee": "पी", "pu": "पु", "poo": "पू",
            "pe": "पे", "pai": "पै", "po": "पो", "pau": "पौ", "pan": "पं", "pam": "पं", "pah": "पः",
            "f": "फ", "fa": "फ", "faa": "फा", "fhi": "िफ", "fee": "फी", "fu": "फु", "foo": "फू",
            "fe": "फे", "fai": "फै", "fo": "फो", "fau": "फौ", "fan": "फं", "fam": "फं", "fah": "फः",
            "b": "ब", "ba": "ब", "baa": "बा", "bi": "िब", "bee": "बी", "bu": "बु", "boo": "बू",
            "be": "बे", "bai": "बै", "bo": "बो", "bau": "बौ", "ban": "बं", "bam": "बं", "bah": "बः",
            "bh": "भ", "bha": "भ", "bhaa": "भा", "bhi": "िभ", "bhee": "भी", "bhu": "भु", "bhoo": "भू",
            "bhe": "भे", "bhai": "भै", "bho": "भो", "bhau": "भौ", "bhan": "भं", "bham": "भं", "bhah": "भः",
            "m": "म", "ma": "म", "maa": "मा", "mi": "िम", "mee": "मी", "mu": "मु", "moo": "मू",
            "me": "मे", "mai": "मै", "mo": "मो", "mau": "मौ", "man": "मं", "mam": "मं", "mah": "मः",
            "y": "य", "ya": "य", "yaa": "या", "yi": "िय", "yee": "यी", "yu": "यु", "yoo": "यू",
            "ye": "ये", "yai": "यै", "yo": "यो", "yau": "यौ", "yan": "यं", "yam": "यं", "yah": "यः",
            "r": "र", "ra": "रा", "raa": "रा", "ri": "िर", "ree": "री", "ru": "रु", "roo": "रू",
            "re": "रे", "rai": "रै", "ro": "रो", "rau": "रौ", "ran": "रं", "ram": "रं", "rah": "रः",
            "l": "ल", "la": "ल", "laa": "ला", "li": "िल", "lee": "ली", "lu": "लु", "loo": "लू",
            "le": "ले", "lai": "लै", "lo": "लो", "lau": "लौ", "lan": "लं", "lam": "लं", "lah": "लः",
            "v": "व", "va": "व", "vaa": "वा", "vi": "िव", "vee": "वी", "vu": "वु", "voo": "वू",
            "ve": "वे", "vai": "वै", "vo": "वो", "vau": "वौ", "van": "वं", "vam": "वं", "vah": "वः",
            "sh": "श", "sha": "श", "shaa": "शा", "shee": "शी", "shu": "शु", "shoo": "शू",
            "she": "शे", "shai": "शै", "sho": "शो", "shau": "शौ", "shan": "शं", "sham": "शं", "shah": "शः",
            "s": "स", "sa": "सा", "saa": "सा", "si": "िस", "see": "सी", "su": "सु", "soo": "सू",
            "se": "से", "sai": "सै", "so": "सो", "sau": "सौ", "sam": "सं", "san": "सं", "sah": "सः",
            "h": "ह", "ha": "हा", "haa": "हा", "hi": "िह", "hee": "ही", "hu": "हु", "hoo": "हू",
            "he": "हे", "hai": "है", "ho": "हो", "hau": "हौ", "han": "हं", "ham": "हं", "hah": "हः",
            "ksh": "क्ष", "ksha": "क्ष", "kshaa": "क्षा", "kshi": "िक्ष", "kshee": "क्षी", "kshu": "क्षु",
            "kshoo": "क्षू", "kshe": "क्षे", "kshai": "क्षै", "ksho": "क्षो", "kshau": "क्षौ", "kshan": "क्षं",
            "ksham": "क्षं", "kshah": "क्षः",
            # Add more complex combinations as needed
            "tr": "त्र", "tra": "त्र", "traa": "त्रा", "tri": "ित्र", "tree": "त्री", "tru": "त्रु",
            "troo": "त्रू", "tre": "त्रे", "trai": "त्रै", "tro": "त्रो", "trau": "त्रौ", "tran": "त्रं",
            "tram": "त्रं", "trah": "त्रः",
            "w": "व", "wa": "व", "waa": "वा",  # w is treated as v in Hindi
            "shw": "श्व", "shwaa": "श्वा",  # Consonant clusters
            "mba": "म्ब", "dm": "द्म", "phi": "िफ़", "phee": "फी", "phu": "फु", "phoo": "फू",
            "phe": "फे", "phai": "फै", "pho": "फो", "phau": "फौ"
        }

        # From ANNEXURE E (Maatra for Transliteration)
        self.hindi_matra_map = {
            "a": "\u093E",  # ा
            "i": "\u093F",  # ि
            "ii": "\u0940",  # ी
            "u": "\u0941",  # ु
            "uu": "\u0942",  # ू
            "r": "\u0943",  # ृ
            "e": "\u0947",  # े
            "ai": "\u0948",  # ै
            "o": "\u094B",  # ो
            "au": "\u094C",  # ौ
            "n": "\u0902",  # ं
        }

        # Common Hindi consonants (without vowel sounds)
        self.hindi_consonants = {
            "k": "क्", "kh": "ख्", "g": "ग्", "gh": "घ्", "ch": "च्", "chh": "छ्", "j": "ज्", "jh": "झ्",
            "t": "ट्", "th": "ठ्", "d": "ड्", "dh": "ढ्", "n": "न्", "p": "प्", "ph": "फ्", "b": "ब्",
            "bh": "भ्", "m": "म्", "y": "य्", "r": "र्", "l": "ल्", "v": "व्", "sh": "श्", "s": "स्", "h": "ह्",
            "tr": "त्र्", "ks": "क्स्", "gn": "ग्न्", "z": "ज़्", "f": "फ़्"
        }

        # Initialize transliteration cache
        self.transliteration_cache = {}

    def load_english_dictionary(self):
        """Load English dictionary from NLTK or create a minimal one."""
        try:
            # Try to use NLTK's words corpus
            from nltk.corpus import words
            self.english_words = set(w.lower() for w in words.words())
            print(f"Loaded {len(self.english_words)} words from NLTK dictionary.")
        except:
            # If NLTK's corpus is not available, use a minimal essential dictionary
            common_words = [
                "so", "awesome", "yummy", "great", "cool", "good", "nice", "love", "hello", "thank",
                "morning", "night", "today", "tomorrow", "please", "thanks", "welcome", "happy",
                "sad", "angry", "amazing", "wonderful", "beautiful", "pretty", "ugly", "smart",
                "dumb", "stupid", "crazy", "insane", "mad", "glad", "sorry", "excuse", "pardon"
            ]
            self.english_words = set(common_words)
            print(f"Using minimal dictionary with {len(self.english_words)} words.")

    def normalize_repeated_chars(self, word):
        """
        Normalize a word with repeated characters by checking against a dictionary.

        Args:
            word (str): Word to normalize

        Returns:
            str: Normalized word
        """
        # Skip short words or words with no repetition
        if len(word) <= 2:
            return word

        # Check if the word has repeated characters (3 or more of the same letter)
        if not re.search(r'([a-zA-Z])\1{2,}', word):
            return word

        # Try both normalization patterns
        single_char = re.sub(r'([a-zA-Z])\1{2,}', r'\1', word)
        double_char = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', word)

        # Check which normalized version exists in our English dictionary
        if single_char.lower() in self.english_words:
            return single_char
        elif double_char.lower() in self.english_words:
            return double_char

        # If neither is in the dictionary, default to single character
        return single_char

    def transliterate_to_devanagari(self, text):
        """
        Transliterate Hindi/Hinglish text in Latin script to Devanagari script
        using character-by-character mapping.

        Args:
            text (str): Hindi/Hinglish text in Latin script

        Returns:
            str: Text in Devanagari script
        """
        # Check if we've already transliterated this word
        if text in self.transliteration_cache:
            return self.transliteration_cache[text]

        # First, normalize the text to handle common variations
        text = self.normalize_word(text.lower())

        # For very short or simple words, try direct mapping first
        if text in self.hindi_to_dev_map:
            self.transliteration_cache[text] = self.hindi_to_dev_map[text]
            return self.hindi_to_dev_map[text]

        # Sort keys by length (descending) to match longer patterns first
        sorted_keys = sorted(self.hindi_to_dev_map.keys(), key=len, reverse=True)

        # Greedy approach: iteratively match and replace longest sequences first
        result = text
        i = 0
        transliterated = ""

        while i < len(text):
            matched = False
            # Try to match the longest possible sequence
            for key_length in range(min(5, len(text) - i), 0, -1):
                substr = text[i:i + key_length]
                if substr in self.hindi_to_dev_map:
                    transliterated += self.hindi_to_dev_map[substr]
                    i += key_length
                    matched = True
                    break

            # If no match found, keep the original character and move forward
            if not matched:
                transliterated += text[i]
                i += 1

        # Store in cache and return
        self.transliteration_cache[text] = transliterated
        return transliterated

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
        """
        Extract character n-gram features from words.
        Uses lemmatization for feature extraction only, not for normalization.
        """
        # Apply lemmatization for feature extraction purposes
        if isinstance(X, np.ndarray):
            X_lemmatized = [self.lemmatize_for_classification(word) for word in X]
        else:
            X_lemmatized = [self.lemmatize_for_classification(word) for word in X]

        if fit:
            self.vectorizer = TfidfVectorizer(
                analyzer='char',
                ngram_range=(1, 4),  # Use character 1-4 grams
                min_df=2,  # Minimum document frequency
                max_features=3000  # Limit features to prevent overfitting
            )
            return self.vectorizer.fit_transform(X_lemmatized)
        else:
            return self.vectorizer.transform(X_lemmatized)

    def lemmatize_for_classification(self, word):
        """
        Apply lemmatization for classification purposes only.
        This doesn't affect the final normalized text output.
        """
        try:
            return self.lemmatizer.lemmatize(word.lower())
        except:
            return word.lower()

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
        Normalize a single word using custom rules.
        Handle English and Hindi words differently.
        Now follows the flowchart more closely with distinct SMS -> English normalization
        steps before language identification.
        """
        # Remove punctuation for normalization, but keep track of it
        word_clean = re.sub(r'[^\w\s]', '', word.lower())
        punctuation = ''.join(c for c in word if c in string.punctuation)
        punct_positions = []
        if word.startswith(punctuation):
            punct_positions.append('start')
        if word.endswith(punctuation):
            punct_positions.append('end')

        # Step 1: Handle wordplay and intentional misspelling (repeated characters)
        word_clean = self.normalize_repeated_chars(word_clean)

        # Step 2: Handle abbreviations and slang words
        # Check if word is in the English normalization dictionary
        normalized = self.english_norm_dict.get(word_clean, word_clean)

        # Step 3: Handle Hindi normalization if it appears to be a Hindi word
        # Very simple heuristic - if it's not in the English dictionary but is in the Hindi one
        if normalized == word_clean:  # If not found in English dictionary
            if word_clean in self.hindi_norm_dict:
                normalized = self.hindi_norm_dict[word_clean]

        # Reattach punctuation if it existed
        if 'start' in punct_positions:
            normalized = punctuation + normalized
        if 'end' in punct_positions:
            normalized = normalized + punctuation

        return normalized

    def normalize_text(self, text):
        """
        Apply comprehensive text normalization following the flowchart.
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

        # Handle contractions, but preserve apostrophes in words like "it's"
        # Instead of replacing "'s" with " is", we'll handle these specially
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'d", " would", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'t", " not", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'m", " am", text)
        # Notably, removed the "'s" -> " is" rule that was causing issues

        # Tokenize
        tokens = word_tokenize(text)

        # Apply word-level normalization based on the flowchart
        normalized_tokens = [self.normalize_word(token) for token in tokens]

        # Join tokens back together
        normalized_text = ' '.join(normalized_tokens)

        return normalized_text

    def tokenize(self, text):
        """Tokenize text into words using NLTK."""
        return word_tokenize(text.lower())

    def process_file(self, input_file, output_dir=None,
                     normalized_file=None, tagged_file=None,
                     metrics_file=None, devanagari_file=None):
        """
        Process input file in four steps:
        1. Normalize text and save to normalized_file
        2. Classify normalized text and save to tagged_file
        3. Transliterate Hindi words to Devanagari script and save to devanagari_file
        4. Calculate metrics between original and normalized text

        Args:
            input_file (str): Path to input file
            output_dir (str, optional): Directory to save output files. If provided, other file paths will be joined with this.
            normalized_file (str, optional): Path to save normalized text. Defaults to "normalized_hinglish.txt".
            tagged_file (str, optional): Path to save tagged text. Defaults to "hinglish_tagged.txt".
            metrics_file (str, optional): Path to save metrics. Defaults to "normalization_metrics.txt".
            devanagari_file (str, optional): Path to save transliterated text. Defaults to "devanagari_output.txt".
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Set default filenames if not provided
        if normalized_file is None:
            normalized_file = "normalized_hinglish.txt"
        if tagged_file is None:
            tagged_file = "hinglish_tagged.txt"
        if metrics_file is None:
            metrics_file = "normalization_metrics.txt"
        if devanagari_file is None:
            devanagari_file = "devanagari_output.txt"

        # Join with output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            normalized_file = os.path.join(output_dir, os.path.basename(normalized_file))
            tagged_file = os.path.join(output_dir, os.path.basename(tagged_file))
            metrics_file = os.path.join(output_dir, os.path.basename(metrics_file))
            devanagari_file = os.path.join(output_dir, os.path.basename(devanagari_file))

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

        # Step 2 & 3: Classify, tag normalized text, and transliterate Hindi words
        final_outputs = []
        with open(tagged_file, 'w', encoding='utf-8') as tag_file, open(devanagari_file, 'w',
                                                                        encoding='utf-8') as dev_file:
            for i, normalized in enumerate(normalized_sentences):
                tag_file.write(f"Sentence [{i + 1}]: {normalized}\n")

                # Split into words and classify
                words = self.tokenize(normalized)
                if not words:
                    continue

                predictions = self.predict(words)

                # Write word-tag pairs and prepare Hindi words for transliteration
                tag_file.write("Word\tLanguage\n")

                # Build the transliterated sentence
                transliterated_words = []
                for word, lang in predictions:
                    tag_file.write(f"{word}\t{lang}\n")

                    # If word is classified as Hindi, transliterate to Devanagari
                    if lang == "HI":
                        transliterated = self.transliterate_to_devanagari(word)
                        transliterated_words.append(transliterated)
                    else:
                        transliterated_words.append(word)

                # Add a blank line in the tagged file
                tag_file.write("\n")

                # Write the mixed English-Devanagari output
                final_output = ' '.join(transliterated_words)
                final_outputs.append(final_output)

                dev_file.write(f"Original [{i + 1}]: {original_sentences[i]}\n")
                dev_file.write(f"Normalized [{i + 1}]: {normalized}\n")
                dev_file.write(f"Mixed E-H [{i + 1}]: {final_output}\n\n")

        # Step 4: Calculate metrics between original and normalized
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
            print(f"With Devanagari: {final_outputs[i]}")
            if i < len(total_bleu_scores):
                print(f"BLEU Score: {total_bleu_scores[i]:.4f}")
            print(f"Edit Distance: {total_edit_distances[i]}\n")

        print(f"Normalized text saved to: {normalized_file}")
        print(f"Tagged text saved to: {tagged_file}")
        print(f"Devanagari output saved to: {devanagari_file}")
        print(f"Metrics saved to: {metrics_file}")

        return normalized_sentences, final_outputs, avg_bleu, avg_edit_distance

    def predict(self, words):
        """Predict language for a list of words."""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained yet. Call train() first or load a model.")

        # Convert input to list if it's a single word
        if isinstance(words, str):
            words = [words]

        # Extract features using lemmatization for classification
        features = self.extract_features(words)

        # Predict
        predictions = self.model.predict(features)

        # Return predictions
        return list(zip(words, predictions))


# Main function
def main():
    # Initialize the classifier
    classifier = HinglishWordClassifier()

    # Create output directories if they don't exist
    output_dir_unofficial = "Normalized output/Unofficial"
    output_dir_official = "Normalized output/Official"

    os.makedirs(output_dir_unofficial, exist_ok=True)
    os.makedirs(output_dir_official, exist_ok=True)

    print(f"Created output directories: {output_dir_unofficial} and {output_dir_official}")

    # Train the model using train.txt
    print("Training model from train.txt...")
    classifier.train(data_file="train.txt")

    # Save the model
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    classifier.save_model(os.path.join(models_dir, "hinglish_model.pkl"))

    # Process input files
    input_file = "hinglish_sentences_unofficial.txt"
    print(f"Processing {input_file}...")
    classifier.process_file(
        input_file=input_file,
        output_dir=output_dir_unofficial
    )

    input_file = "hinglish_sentences_official.txt"
    print(f"Processing {input_file}...")
    classifier.process_file(
        input_file=input_file,
        output_dir=output_dir_official
    )


if __name__ == "__main__":
    main()