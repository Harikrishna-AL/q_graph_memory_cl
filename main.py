import argparse
from src.data_utils import get_dataloader
from src.model import load_dino, extract_features
from src.learner import train_continual_graph
from src.evaluators import evaluate_graph
from src.config import Config

def main():
    parser = argparse.ArgumentParser(description="Graph Memory Continual Learning")
    parser.add_argument("--use_train", action="store_true", default=True, help="Use full TRAIN set (Recommended)")
    parser.add_argument("--words", type=int, default=512, help="Words per task per chunk")
    args = parser.parse_args()
    
    # Update Config dynamically
    Config.WORDS_PER_TASK = args.words
    
    print("=== 🧠 Graph Memory Project ===")
    
    # 1. Data
    dataset, loader = get_dataloader(use_train_set=args.use_train)
    
    # 2. Features
    dino = load_dino()
    features, labels = extract_features(dino, loader)
    
    # 3. Train
    graph, test_feats, test_lbls = train_continual_graph(features, labels)
    
    # 4. Eval
    evaluate_graph(graph, test_feats, test_lbls)

if __name__ == "__main__":
    main()