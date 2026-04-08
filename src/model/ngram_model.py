import json
import logging
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class NGramModel:
    """Building, storing, and exposing n-gram probability tables with backoff."""
    
    def __init__(self, n_order: int = 4):
        self.n_order = n_order
        self.vocab = set()
        self.model = {f"{i}gram": defaultdict(Counter) for i in range(1, n_order + 1)}
        self.probs = {f"{i}gram": {} for i in range(1, n_order + 1)}

    def build_vocab(self, token_file: str, unk_threshold: int):
        """Builds vocabulary and applies UNK threshold."""
        word_counts = Counter()
        with open(token_file, 'r') as f:
            for line in f:
                word_counts.update(line.split())
        
        self.vocab = {word for word, count in word_counts.items() if count >= unk_threshold}
        self.vocab.add("<UNK>")
        logger.info(f"Vocab built. Size: {len(self.vocab)}")

    def build_counts_and_probabilities(self, token_file: str):
        """Counts n-grams 1..N and computes MLE probabilities."""
        total_words = 0
        
        with open(token_file, 'r') as f:
            for line in f:
                tokens = [t if t in self.vocab else "<UNK>" for t in line.split()]
                total_words += len(tokens)
                
                for n in range(1, self.n_order + 1):
                    for i in range(len(tokens) - n + 1):
                        ngram = tokens[i:i+n]
                        context = " ".join(ngram[:-1])
                        target = ngram[-1]
                        self.model[f"{n}gram"][context][target] += 1

        # Compute Probabilities
        for n in range(1, self.n_order + 1):
            for context, counts in self.model[f"{n}gram"].items():
                total_context = sum(counts.values())
                self.probs[f"{n}gram"][context] = {
                    word: count / total_context for word, count in counts.items()
                }
        logger.info("Probabilities computed for all orders.")

    def lookup(self, context_list: list[str]) -> dict:
        """Backoff lookup: highest order down to 1-gram."""
        for n in range(self.n_order, 0, -1):
            needed_len = n - 1
            current_context = " ".join(context_list[-needed_len:]) if needed_len > 0 else ""
            
            if current_context in self.probs[f"{n}gram"]:
                logger.debug(f"Match found at {n}gram for context: '{current_context}'")
                return self.probs[f"{n}gram"][current_context]
        
        return {}

    def save_model(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.probs, f)

    def save_vocab(self, path: str):
        with open(path, 'w') as f:
            json.dump(list(self.vocab), f)

    def load(self, model_path: str, vocab_path: str):
        try:
            with open(model_path, 'r') as f:
                self.probs = json.load(f)
            with open(vocab_path, 'r') as f:
                self.vocab = set(json.load(f))
        except FileNotFoundError:
            logger.error("model.json not found. Run --step model first.")
            raise
        except json.JSONDecodeError:
            logger.error("model.json is malformed. Re-run --step model.")
            raise