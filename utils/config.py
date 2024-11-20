import argparse

def csv_list(string):
    # convert csv row to a list
    return string.split(',')

def parse_args_llama():
    """
    get llama args
    """
    parser = argparse.ArgumentParser(description="G-Retreiver")

    # model name, project name, and seed
    parser.add_argument("--model_name", type=str, default="graph_llm")
    parser.add_argument("--project", type=str, default="project_g_retriever")
    parser.add_argument("--seed", type=int, default=0)

    # dataset, learn rate, weight decay, patience
    parser.add_argument("--dataset", type=str, default="expla_graphs")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--patience", type=float, default=2)

    # model training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_steps", type=int, default=2)

    # learning rate scheduler
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=int, default=1)

    # inference
    parser.add_argument("--eval_batch_size", type=int, default=16)

    # llm related
    parser.add_argument("--llm_model_name", type=str, deafult='7b')
    parser.add_argument("--llm_model_path", type=str, default='')
    parser.add_argument("--llm_frozen", type=str, default='True')
    parser.add_argument("--llm_num_virtual_tokens", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default='output')
    parser.add_argument("--max_txt_len", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_memory", type=csv_list, default=[80,80])
    
    # gnn related
    parser.add_argument("--gnn_model_name", type=str, default='gt')
    parser.add_argument("--gnn_num_layers", type=int, default=4)
    parser.add_argument("--gnn_in_dim", type=int, default=1024)
    parser.add_argument("--gnn_hidden_dim", type=int, default=1024)
    parser.add_argument("--gnn_num_heads", type=int, default=4)
    parser.add_argument("--gnn_dropout", type=float, default=0.0)
    

    # get args we just initialized and return them
    args = parser.parse_args()
    return args