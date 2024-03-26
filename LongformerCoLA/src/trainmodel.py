from utils.train import train_model
import argparse

def main(args):
    config = {
        'ds_dir': args.ds_dir,
        'batch_size': args.batch_size,
        'num_layers': args.num_layers,
        'd_model': args.d_model,
        'num_heads': args.num_heads,
        'intermediate_size': args.intermediate_size,
        'num_embeddings': args.num_embeddings,
        'dropout': args.dropout,
        'num_labels': args.num_labels,
        'lr': args.lr,
        'epochs': args.epochs,
        'device': args.device,
        'model_folder' : "weights",
        "model_basename" : "lmodel_",
        "preload" : None,
        "tokenizer_file" : "tokenizer_work.json",
        "experiment_name" : "runs/lmodel",
        'seq_len' : args.seq_len
    }
    train_model(config)

def test(args):
    config = {
        'ds_dir': args.ds_dir,
        'batch_size': args.batch_size,
        'num_layers': args.num_layers,
        'd_model': args.d_model,
        'num_heads': args.num_heads,
        'intermediate_size': args.intermediate_size,
        'num_embeddings': args.num_embeddings,
        'dropout': args.dropout,
        'num_labels': args.num_labels,
        'lr': args.lr,
        'epochs': args.epochs,
        'device': args.device,
        'model_folder' : "weights",
        "model_basename" : "lmodel_",
        "preload" : None,
        "tokenizer_file" : "tokenizer_work.json",
        "experiment_name" : "runs/lmodel",
        'seq_len' : args.seq_len
    }
    return config
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_dir', type=str, required=False, default='./Files/spam.csv')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--intermediate_size', type=int, default=3072)
    parser.add_argument('--num_embeddings', type=int, default=30522)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--device', type=str, required=False, default='cuda')
    parser.add_argument('--seq_len', type= int, required=False, default= 512)
    arguments = parser.parse_args()
    main(arguments)