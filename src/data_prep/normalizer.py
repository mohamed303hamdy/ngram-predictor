import os
import re
import logging
import unicodedata

logger = logging.getLogger(__name__)

class Normalizer:
    """Responsible for loading, cleaning, tokenizing, and saving the corpus."""
    
    def __init__(self):
        self.punctuation_regex = re.compile(r'[^\w\s]')
        self.number_regex = re.compile(r'\d+')

    def load(self, folder_path: str) -> str:
        """Loads all .txt files from a folder and concatenates them."""
        combined_text = []
        try:
            files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
            if not files:
                logger.warning(f"No text files found in {folder_path}")
            for filename in files:
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8-sig') as f:
                    combined_text.append(f.read())
            return "\n".join(combined_text)
        except FileNotFoundError:
            logger.error(f"Folder not found: {folder_path}. Check TRAIN_RAW_DIR in .env")
            raise

    def strip_gutenberg(self, text: str) -> str:
        """Removes Project Gutenberg headers and footers using markers."""
        start_marker = re.compile(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .* \*\*\*")
        end_marker = re.compile(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .* \*\*\*")
        
        start_match = start_marker.search(text)
        end_match = end_marker.search(text)
        
        content_start = start_match.end() if start_match else 0
        content_end = end_match.start() if end_match else len(text)
        
        return text[content_start:content_end]

    def normalize(self, text: str) -> str:
        """lowercase → remove punctuation → remove numbers → remove whitespace."""
        text = text.lower()
        text = self.punctuation_regex.sub('', text)
        text = self.number_regex.sub('', text)
        return " ".join(text.split())

    def sentence_tokenize(self, text: str) -> list[str]:
        """Splits text into sentences based on newlines/punctuation logic."""
        # Simplified for this project: Gutenberg sentences usually end with . ! ?
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def word_tokenize(self, sentence: str) -> list[str]:
        """Splits a single sentence into a list of normalized tokens."""
        clean_sentence = self.normalize(sentence)
        return clean_sentence.split()

    def save(self, sentences: list[list[str]], filepath: str):
        """Writes tokenized sentences to output file: one sentence per line."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            for tokens in sentences:
                f.write(" ".join(tokens) + "\n")