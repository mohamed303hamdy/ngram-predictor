import math
import logging
from src.model.ngram_model import NGramModel
from src.data_prep.normalizer import Normalizer

logger = logging.getLogger(__name__)

class Evaluator:
    """Computes perplexity on a held-out evaluation corpus."""
    
    def __init__(self, model: NGramModel, normalizer: Normalizer):
        self.model = model
        self.normalizer = normalizer

    def score_word(self, word: str, context: list[str]) -> float:
        """Returns log2 P(word | context). Returns None if prob is 0."""
        # Map context to OOV
        clean_context = [w if w in self.model.vocab else "<UNK>" for w in context]
        target = word if word in self.model.vocab else "<UNK>"
        
        predictions = self.model.lookup(clean_context)
        prob = predictions.get(target, 0.0)
        
        return math.log2(prob) if prob > 0 else None

    def compute_perplexity(self, eval_tokens_path: str) -> dict:
        """Calculates Perplexity: 2^(-1/N * sum(log2(P)))"""
        total_log_prob = 0.0
        word_count = 0
        skipped = 0
        
        with open(eval_tokens_path, 'r') as f:
            for line in f:
                tokens = line.split()
                for i in range(len(tokens)):
                    word = tokens[i]
                    context = tokens[max(0, i - (self.model.n_order - 1)):i]
                    
                    score = self.score_word(word, context)
                    if score is not None:
                        total_log_prob += score
                        word_count += 1
                    else:
                        skipped += 1
        
        if word_count == 0: return {"perplexity": float('inf'), "count": 0, "skipped": skipped}
        
        cross_entropy = - (total_log_prob / word_count)
        perplexity = 2 ** cross_entropy
        
        if (skipped / (word_count + skipped)) > 0.20:
            logger.warning(f"High skip rate: {skipped} words had zero probability.")
            
        return {
            "perplexity": round(perplexity, 2),
            "count": word_count,
            "skipped": skipped
        }

    def run(self, eval_file: str):
        """Orchestrates the evaluation and prints results."""
        results = self.compute_perplexity(eval_file)
        print(f"\n--- Evaluation Results ---")
        print(f"Perplexity: {results['perplexity']}")
        print(f"Words evaluated: {results['count']}")
        print(f"Words skipped: {results['skipped']}")