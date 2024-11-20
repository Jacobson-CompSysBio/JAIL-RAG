import os
import torch

# trainable parms fn. to check if frozen
def print_trainable_params(model):
    # init param counter
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        # get num params for param groups
        num_params = param.numel()

        # add to total
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    return trainable_params, all_param

def _save_checkpoint(model, optimizer, cur_epoch, args, is_best=False):
    """
    Save checkpoint at the current epoch
    """

    # make directory if it doesn't exist
    os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)

    # get dict of trainable params
    param_grad_dic = {
        k: v.requires_grad for k, v in model.named_parameters()
    }

    # get state dict, but only keep params that require grad
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            del state_dict[k]
    
    # make dict of things to save
    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "config": args,
        "epoch": cur_epoch,
    }

    # get path and save
    # gotta be a better name for this path
    # path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_'

    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, path))
    torch.save(save_obj, path)

def _reload_best_model(model, args):
    """
    Load best ckpt for evaluation
    """

    # make a better path
    checkpoint_path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_best.pth'
    print(f"Loading checkpoint from {checkpoint_path}.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    return model

def _reload_model(model, checkpoint_path):
    """
    Load model from checkpoint paths
    """

    print(f"Loading checkpoint from {checkpoint_path}.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    return model