import logging
import regex as re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import contractions
from textblob import TextBlob
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Predefined important phrases to preserve
important_phrases = {
    'Not interested': 'Not interested',
    'No Vaccine': 'No Vaccine',
    'No Response': 'No Response'
}

class PreserveImportant:
# Replaces important phrases with placeholders to preserve them during text processing.
    def __call__(self, text):
        try:
            for phrase in important_phrases.keys():
                placeholder = self._generate_placeholder(phrase)
                text = re.sub(r'\b' + re.escape(phrase) + r'\b', placeholder, text)
            return text
        except Exception as e:
            logging.error(f"Error in PreserveImportant: {e}")
            return text

    @staticmethod
    def _generate_placeholder(phrase):
        return f'__{phrase.replace(" ", "_")}__'

class RestoreImportant:
# Restores preserved placeholders back to their original phrases.
    def __call__(self, text):
        try:
            for phrase in reversed(important_phrases.keys()):  # Reverse to handle nested placeholders
                placeholder = PreserveImportant._generate_placeholder(phrase)
                text = text.replace(placeholder, phrase)
            return text
        except Exception as e:
            logging.error(f"Error in RestoreImportant: {e}")
            return text

class LowerCase:
# Converts text to lowercase.
    def __call__(self, text):
        try:
            return text.lower()
        except Exception as e:
            logging.error(f"Error in LowerCase: {e}")
            return text

class RemovePunctuations:
# Removes punctuation marks from text.
    def __init__(self):
        self.punctuations = string.punctuation

    def __call__(self, text):
        try:
            return text.translate(str.maketrans('', '', self.punctuations))
        except Exception as e:
            logging.error(f"Error in RemovePunctuations: {e}")
            return text

class Tokenize:
# Splits text into tokens.
    def __init__(self, tokenize_fn=None):
        self.tokenize_fn = tokenize_fn or str.split

    def __call__(self, text):
        try:
            return self.tokenize_fn(text)
        except Exception as e:
            logging.error(f"Error in Tokenize: {e}")
            return text

class RemoveStopWords:
# Removes common stopwords from text.
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            logging.warning("NLTK stopwords not found. Falling back to an empty set.")
            self.stop_words = set()

    def __call__(self, text):
        try:
            words = text.split(' ')
            filtered_words = [word for word in words if word.lower() not in self.stop_words]
            return ' '.join(filtered_words)
        except Exception as e:
            logging.error(f"Error in RemoveStopWords: {e}")
            return text

class Stem:
# Applies stemming to words in the text.
    def __init__(self, stemmer=PorterStemmer()):
        self.stemmer = stemmer

    def __call__(self, text):
        try:
            words = text.split(' ')
            stemmed = [self.stemmer.stem(word) for word in words]
            return ' '.join(stemmed)
        except Exception as e:
            logging.error(f"Error in Stem: {e}")
            return text

class Lemmatize:
# Lemmatizes words based on their part-of-speech tags.
    def __init__(self, lemmatizer=None):
        self.lemmatizer = lemmatizer or WordNetLemmatizer()

    def __call__(self, text):
        try:
            words = word_tokenize(text)
            pos_tagged = pos_tag(words)
            lemmatized = [
                self.lemmatizer.lemmatize(word, self.get_wordnet_pos(pos)) or word
                for word, pos in pos_tagged
            ]
            return ' '.join(lemmatized)
        except Exception as e:
            logging.error(f"Error in Lemmatize: {e}")
            return text

    @staticmethod
    def get_wordnet_pos(treebank_tag):
# Converts POS tags to WordNet format.
        try:
            if treebank_tag.startswith('J'):
                return wordnet.ADJ
            elif treebank_tag.startswith('V'):
                return wordnet.VERB
            elif treebank_tag.startswith('N'):
                return wordnet.NOUN
            elif treebank_tag.startswith('R'):
                return wordnet.ADV
            return wordnet.NOUN
        except Exception as e:
            logging.error(f"Error in get_wordnet_pos: {e}")
            return wordnet.NOUN

class RemoveSpecialCharacters:
# Removes special characters from text.
    def __init__(self, pattern=None):
        self.pattern = pattern or r'[^a-zA-Z0-9\s]'

    def __call__(self, text):
        try:
            return re.sub(self.pattern, '', text)
        except Exception as e:
            logging.error(f"Error in RemoveSpecialCharacters: {e}")
            return text

class RemoveEmojis:
# Removes emojis from text.
    def __call__(self, text):
        try:
            emoji_pattern = re.compile(
               "["
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
            "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
            "\U0001F700-\U0001F77F"  # Alchemical Symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251"  # Enclosed Characters
            "]+", flags=re.UNICODE
            )
            return emoji_pattern.sub(r'', text)
        except Exception as e:
            logging.error(f"Error in RemoveEmojis: {e}")
            return text

class HandleMispellings:
# Corrects common misspellings.
    def __call__(self, text):
        try:
            words = text.split(' ')
            corrected = [str(TextBlob(word).correct()) for word in words]
            return ' '.join(corrected)
        except Exception as e:
            logging.error(f"Error in HandleMispellings: {e}")
            return text

class HandleContractions:
# Expands contractions in text.
    def __call__(self, text):
        try:
            return contractions.fix(text)
        except Exception as e:
            logging.error(f"Error in HandleContractions: {e}")
            return text

class Compose:
# Chains multiple text preprocessing steps together.
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, text):
        try:
            for transform in self.transforms:
                text = transform(text)
            return text
        except Exception as e:
            logging.error(f"Error in Compose: {e}")
            return text

class TextPreprocessor:
# Main class to preprocess text using a predefined pipeline.
    def __init__(self):
        self.transforms = Compose([
            PreserveImportant(),
            LowerCase(),
            HandleMispellings(),
            HandleContractions(),
            RemovePunctuations(),
            RemoveSpecialCharacters(),
            RemoveEmojis(),
            RemoveStopWords(),
            Lemmatize(),
            RestoreImportant(),
        ])

    def preprocess_response(self, response):
        try:
            if pd.isnull(response):
                return ''
            return self.transforms(response)
        except Exception as e:
            logging.error(f"Error in preprocess_response: {e}")
            return response