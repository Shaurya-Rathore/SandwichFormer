from utils.train import train_model
import argparse
from utils.config import config

def main(args):
    config = {
        'ds_dir': args.ds_dir,
        'batch_size': args.batch_size,
        'num_layers': args.num_layers,
        'hidden_size': args.hidden_size,
        'num_attention_heads': args.num_attention_heads,
        'intermediate_size': args.intermediate_size,
        'num_embeddings': args.num_embeddings,
        'dropout': args.dropout,
        'num_classes': args.num_classes,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'device': args.device
    }
    train_model(config)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_dir', type=str, required=False, default='./Files/spam.csv')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_attention_heads', type=int, default=12)
    parser.add_argument('--intermediate_size', type=int, default=3072)
    parser.add_argument('--num_embeddings', type=int, default=30522)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=2e-9)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--device', type=str, required=False, default='cuda')
    arguments = parser.parse_args()
    main(arguments)