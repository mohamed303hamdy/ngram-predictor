import logging

logger = logging.getLogger(__name__)

class Predictor:
    """Orchestrates normalization, OOV mapping, and model lookup."""
    
    def __init__(self, model, normalizer, n_order: int):
        self.model = model
        self.normalizer = normalizer
        self.n_order = n_order

    def map_oov(self, context: list[str]) -> list[str]:
        """Replaces words not in vocabulary with <UNK>."""
        return [w if w in self.model.vocab else "<UNK>" for w in context]

    def predict_next(self, text: str, k: int) -> list:
        """Predicts the next k words based on input text."""
        if not text.strip():
            logger.warning("Empty input string provided.")
            return []
            
        normalized_tokens = self.normalizer.word_tokenize(text)
        context = self.map_oov(normalized_tokens)
        
        predictions = self.model.lookup(context)
        # Sort by probability descending
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return [word for word, prob in sorted_preds[:k]]