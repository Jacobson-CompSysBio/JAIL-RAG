# code from meta
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

def adjust_learning_rate(param_group, LR, epoch, args):
    """
    Decay the learning rate with half-cycle cosine after warmup.
    """

    # pull out args (probably don't need them, going to check the training script)
    num_epochs = args.num_epochs
    warmup_epochs = args.warmup_epochs
    min_lr = 5e-6

    # apply warmup
    if epoch < warmup_epochs:
        lr = LR * epoch / warmup_epochs
    
    # if outside of warmup, apply cosine decay
    else:
        lr = min_lr + (LR - min_lr) * 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    
    # apply lr
    param_group["lr"] = lr
    
    return lr