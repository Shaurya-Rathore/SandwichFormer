from pathlib import Path
def get_config():
    return {
        "batch_size": 4,
        "epochs": 2,
        "lr": 1e-3,
        "d_model": 768,
        "seq_len": 512,
        "num_layers": 12,
        "num_heads": 12,
        "dropout" : 0.1,
        "intermediate_size" : 3072,
        "num_labels": 2,
        "model_folder" : "weights",
        "model_basename" : "lmodel_",
        "preload" : None,
        "tokenizer_file" : "tokenizer_work.json",
        "experiment_name" : "runs/lmodel"
    }
    
def get_weights_path(config, epoch):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)
