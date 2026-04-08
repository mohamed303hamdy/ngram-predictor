import pytest
import os
from src.model.ngram_model import NGramModel

def test_build_vocab_with_unk():
    # Setup dummy tokens
    with open("test_tokens.txt", "w") as f:
        f.write("apple apple apple banana cherry")
    
    model = NGramModel(n_order=2)
    model.build_vocab("test_tokens.txt", unk_threshold=2)
    
    assert "apple" in model.vocab
    assert "banana" not in model.vocab
    assert "<UNK>" in model.vocab
    os.remove("test_tokens.txt")

def test_backoff_logic():
    model = NGramModel(n_order=2)
    model.vocab = {"i", "am", "sherlock"}
    # Manually inject probabilities for a Bigram
    model.probs["2gram"]["i"] = {"am": 1.0}
    model.probs["1gram"][""] = {"sherlock": 0.5}
    
    # Test seen context
    assert "am" in model.lookup(["i"])
    # Test unseen context (should backoff to unigram)
    assert "sherlock" in model.lookup(["he"])