import os
import argparse
import logging 
import sys
from dotenv import load_dotenv

# Import our custom modules
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor

def setup_logging():
    """Configures logging based on .env settings."""
    load_dotenv("config/.env")
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

def main():
    setup_logging()
    
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="Sherlock Holmes N-Gram Predictor")
    parser.add_argument("--step", 
                        choices=["dataprep", "model", "inference", "evaluate", "all"], 
                        required=True,
                        help="The pipeline step to execute")
    args = parser.parse_args()

    # 2. Dependency Injection - Instantiate shared objects
    norm = Normalizer()
    ngram_order = int(os.getenv("NGRAM_ORDER", 4))
    model = NGramModel(n_order=ngram_order)
    
    # --- STEP: DATAPREP ---
    if args.step in ["dataprep", "all"]:
        logging.info("Starting Data Preparation...")
        raw_text = norm.load(os.getenv("TRAIN_RAW_DIR"))
        clean_text = norm.strip_gutenberg(raw_text)
        sentences = norm.sentence_tokenize(clean_text)
        tokenized_data = [norm.word_tokenize(s) for s in sentences]
        norm.save(tokenized_data, os.getenv("TRAIN_TOKENS"))
        logging.info("Data Preparation complete.")

    # --- STEP: MODEL ---
    if args.step in ["model", "all"]:
        logging.info("Starting Model Training...")
        model.build_vocab(os.getenv("TRAIN_TOKENS"), int(os.getenv("UNK_THRESHOLD", 3)))
        model.build_counts_and_probabilities(os.getenv("TRAIN_TOKENS"))
        model.save_model(os.getenv("MODEL"))
        model.save_vocab(os.getenv("VOCAB"))
        logging.info("Model training complete.")

    # --- STEP: EVALUATE (Extra Credit) ---
    if args.step == "evaluate":
        logging.info("Starting Evaluation...")
        from src.evaluation.evaluator import Evaluator
        
        # Ensure model is loaded before evaluating
        model.load(os.getenv("MODEL"), os.getenv("VOCAB"))
        evaluator = Evaluator(model, norm)
        
        # Process the evaluation corpus
        raw_eval = norm.load(os.getenv("EVAL_RAW_DIR"))
        clean_eval = norm.strip_gutenberg(raw_eval)
        sentences_eval = norm.sentence_tokenize(clean_eval)
        tokenized_eval = [norm.word_tokenize(s) for s in sentences_eval]
        norm.save(tokenized_eval, os.getenv("EVAL_TOKENS"))
        
        evaluator.run(os.getenv("EVAL_TOKENS"))

    # --- STEP: INFERENCE ---
    if args.step in ["inference", "all"]:
        logging.info("Entering Inference Mode...")
        model.load(os.getenv("MODEL"), os.getenv("VOCAB"))
        predictor = Predictor(model, norm, ngram_order)
        k = int(os.getenv("TOP_K", 3))
        
        print("\n" + "="*40)
        print("Sherlock Predictor CLI")
        print("Type your phrase and press Enter.")
        print("Type 'quit' to exit.")
        print("="*40)

        while True:
            try:
                user_input = input("\n> ")
                if user_input.lower() in ['quit', 'exit']: 
                    break
                if not user_input.strip():
                    continue
                
                preds = predictor.predict_next(user_input, k)
                print(f"Predictions: {preds}")
            except KeyboardInterrupt:
                break
        print("\nGoodbye.")

if __name__ == "__main__":
    main()