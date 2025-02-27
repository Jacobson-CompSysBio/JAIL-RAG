import os, sys, glob
import re
import numpy as np
import torch

sys.path.append('../')

# define reward function for node connectivity
def node_connectivity_reward(gt, pred) -> int: 

    # extract answer from prediction
    ans = ''.join(re.findall(r"<answer>(.*?)</answer>", pred)) 
    ans = ans.lower() 

    # if the model didn't produce an answer, return -1
    if ans == "":
        return -1
    
    # if the model produced an answer, compare it to the ground truth - return 1 if correct, -1 if incorrect
    if ans == gt:
        return 1
    else:
        return -1
    
