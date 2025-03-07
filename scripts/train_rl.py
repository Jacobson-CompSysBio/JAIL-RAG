# Import necessary libraries
# Basic Python libraries for various operations
import random
import copy
import re
import os
import sys
import numpy as np
import wandb
from dotenv import load_dotenv
from DGXutils import GetLowestGPU
from tqdm.auto import tqdm

# PyTorch and related libraries for deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# Hugging Face libraries for transformer models
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.append('../')

# custom
from utils import preprocess as pp
from utils.graph_llm import GraphLLM
from utils.llm import LLM
from utils.multiplex import Multiplex
from utils.textualize import *
from utils.bio_graphs import BiologicalDataset
from utils.evaluate import eval_funcs
from utils.config import parse_args_llama
from utils.ckpt import _save_checkpoint, _reload_best_model
from utils.collate import collate_fn
from utils.seed import seed_everything
from utils.lr_schedule import adjust_learning_rate

def set_random_seed(seed: int = 42):
    """
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use for random number generation.

    Returns:
        None

    Explanation:
        1. Sets seed for Python's built-in random module for basic random operations.
        2. Sets seed for NumPy, ensuring consistent random number generation in array operations.
        3. Sets seed for PyTorch CPU operations.
        4. If CUDA is available, sets seed for all GPU devices.
        5. Configures cuDNN to ensure deterministic behavior:
           - Sets deterministic flag to True, ensuring reproducible results.
           - Disables benchmarking to prevent algorithm selection based on hardware.

    Note:
        Setting deterministic behavior may impact performance but ensures consistent results
        across multiple runs, which is crucial for debugging and research.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call the function to set random seed for reproducibility
set_random_seed(42)

load_dotenv()
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT")
os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY")

# set visible devices to gpus 0-3
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

T = 512

# get dataset to see what we're working with
path = "../data/subgraphs/all/"
dataset = BiologicalDataset(path)
loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# load model to see what we're working with
model = GraphLLM(max_txt_len=T,
                max_new_tokens=200,
                llm_model_path='meta-llama/Meta-Llama-3.1-8B-Instruct',
                llm_frozen=False, # set frozen to false so we can train with RL
                fsdp=False, 
                )

def extract_answer(text):
    """
    Extract answer from the model output.

    Args:
        text (str): The model output text.
    
    Returns:
        str: The extracted answer.
    """
    
    # extract answer from prediction
    ans = ''.join(re.findall(r"<answer>(.*?)</answer>", text)[-1]) 
    ans = ans.lower() 
    return ans


def evaluate_model(model, batch):
    """
    Evaluate the model on a set of examples provided by a PyTorch DataLoader.

    Args:
        model (GraphLLM): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): A DataLoader yielding evaluation batches.
            Each batch is expected to be a dictionary with keys such as 'id', 'question', 
            'scope', 'label', 'desc', and 'graph'. The values for 'label', 'desc', and 'question'
            should be lists (or tensors in the case of labels) of the same batch size.

    Returns:
        float: The accuracy of the model on the evaluation examples.
    
    References:
        - PyTorch DataLoader documentation: https://pytorch.org/docs/stable/data.html
        - Accelerate library for device placement and distributed inference: https://huggingface.co/docs/accelerate 
    """
    model.eval()
    correct = 0

    batch_size = len(batch["label"])
    print("\n" + "=" * 50)
    print(f"EVALUATION ON {batch_size} EXAMPLES")
    print("=" * 50)

    # Perform model inference on the whole batch with no gradient computation.
    with torch.no_grad():
        outputs = model.inference(batch)

    # Assume outputs["pred"] is a list or tensor of predictions of length equal to batch_size.
    for i in range(batch_size):
        # Extract the predicted answer for this example.
        predicted = extract_answer(outputs["pred"][i])
        expected = batch["label"][i]
        is_correct = (predicted == expected)
        if is_correct:
            correct += 1

        # Print details for this example.
        print("\nPrompt:")
        print(batch["desc"][i] + ' ' + batch["question"][i])
        print("\nExpected Answer:")
        print(expected)
        print("\nExtracted Answer:")
        print(predicted)
        print("\nFull Generated Response:")
        # If outputs["pred"] is a tensor or list of strings, print accordingly.
        print(outputs["pred"][i])
        print("\nCorrect:", "✓" if is_correct else "✗")
        print("-" * 50)

    accuracy = (correct / batch_size) * 100
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{batch_size})")
    print("=" * 50)

    # Switch model back to training mode after evaluation.
    model.train()
    return accuracy


# function to reward formatting
def reward_format(gt, pred):
    """
    if the answer is in the correct format, reward 1.25, else reward -1
    """
    
    # answer format
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"

    return 1.25 if re.match(pattern, pred, re.DOTALL | re.VERBOSE) else -1

# define reward function for node connectivity
def reward_correct_yn(gt, pred) -> int: 
    """
    given a yes/no answer and ground truth, return 1 if correct, -1 if incorrect
    """

    # extract answer from prediction
    ans = ''.join(re.findall(r"<answer>(.*?)</answer>", pred)) 
    ans = ans.lower() 

    # if the model produced an answer, compare it to the ground truth - return 1 if correct, -1 if incorrect
    if ans == gt:
        return 1
    else:
        return -1
    
def combined_reward(prompts, completions, answer):
    """
    Combined reward function for yes/no questions and answer formatting.
    
    Args:
        prompts (list): List of prompt strings (unused in this example, but available for context).
        completions (list): List where each element is a list of dictionaries representing generated completions.
                            Each dictionary should have a key 'content' with the generated text.
        answer (list): List of ground truth answers.
        
    Returns:
        list: A list of reward values computed for each sample.
    """
    rewards = []
    # Loop over each sample's ground truth and generated completions.
    for gt, comp in zip(answer, completions):
        # For simplicity, use the first completion for computing the reward.
        pred = comp[0]['content']
        r1 = reward_correct_yn(gt, pred)
        r2 = reward_format(gt, pred)
        rewards.append(r1 + r2)
    return rewards

def selective_log_softmax(logits, input_ids):
    """
    Computes log probabilities for specific tokens in the vocabulary.

    Args:
        logits (torch.Tensor): The raw logits output from the model.
        input_ids (torch.Tensor): The token IDs for which we want the log probabilities.

    Returns:
        torch.Tensor: Log probabilities of the selected tokens.

    Explanation:
        1. Applies log softmax to convert logits to log probabilities over the vocabulary.
        2. Uses gather to extract only the log probabilities corresponding to the input_ids.
        3. Removes the extra dimension to match the original shape of input_ids.
    """
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

def compute_log_probs(model, input_ids, attention_mask, logits_to_keep):
    """
    Computes log probabilities for the last `logits_to_keep` tokens using the underlying transformer model.

    Args:
        model (GraphLLM): The GraphLLM instance.
        input_ids (torch.Tensor): Tensor of input token IDs.
        attention_mask (torch.Tensor): Attention mask corresponding to the input IDs.
        logits_to_keep (int): Number of tokens (from the end of the sequence) to compute log probabilities for.

    Returns:
        torch.Tensor: Log probabilities for the selected tokens.
    """
    # Call the underlying transformer model directly, since it accepts input_ids and attention_mask.
    outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Slice to get the logits for the last `logits_to_keep` tokens.
    logits = outputs.logits[:, -logits_to_keep:, :]
    
    # Also slice the input_ids to match the logits.
    trimmed_input_ids = input_ids[:, -logits_to_keep:]
    
    # Use your custom selective log softmax to compute log probabilities.
    return selective_log_softmax(logits, trimmed_input_ids)

def create_completion_mask(completion_ids, eos_token_id):
    """
    Creates a mask for completion tokens that excludes tokens after the EOS token.

    Args:
        completion_ids (torch.Tensor): Token IDs of the generated completions.
        eos_token_id (int): The ID of the end-of-sequence token.

    Returns:
        torch.Tensor: A binary mask with 1s for valid tokens and 0s after the EOS token.

    Explanation:
        1. Identifies positions where EOS tokens occur in each sequence.
        2. Finds the index of the first EOS token in each sequence.
        3. Creates a mask where positions before and including the first EOS are 1, others are 0.
        4. If no EOS token is found in a sequence, all positions are set to 1.
    """
    is_eos = completion_ids == eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
    mask_exists = is_eos.any(dim=1)
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
    sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
    return (sequence_indices <= eos_idx.unsqueeze(1)).int()

def generate_completions(model, batch, num_generations=4, max_completion_length=32):
    """
    Generates multiple completions for each prompt.

    Args:
        model: The language model.
        tokenizer: The tokenizer for encoding and decoding text.
        prompts (list): List of text prompts.
        num_generations (int): Number of completions to generate per prompt.
        max_completion_length (int): Maximum number of tokens to generate.

    Returns:
        tuple: Containing prompt IDs, prompt mask, completion IDs, and completion mask.

    Explanation:
        1. Encodes the prompts and moves them to the appropriate device.
        2. Repeats each prompt num_generations times to generate multiple completions.
        3. Generates completions using the model with specified parameters.
        4. Extracts the completion IDs (excluding the prompt tokens).
        5. Creates a mask for the completions using create_completion_mask.
    """

    # tokenize prompt inputs
    prompt_inputs = [batch["desc"][i] + batch["question"][i] for i in range(len(batch["desc"]))]
    inputs = model.tokenizer(prompt_inputs, return_tensors="pt", padding=True, padding_side="left")
    prompt_ids = inputs["input_ids"]
    prompt_mask = inputs["attention_mask"]
    print(f"Input batch size: {prompt_ids.size(0)}")

    prompt_length = prompt_ids.size(1)
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)

    outputs = model.inference(batch, num_generations=num_generations)

    print(f"Output batch size: {outputs['out_ids'].size(0)}")
    completion_ids = outputs["out_ids"][:, prompt_length:]
    completion_mask = create_completion_mask(completion_ids, model.tokenizer.eos_token_id)
    return prompt_ids, prompt_mask, completion_ids, completion_mask

def generate_rollout_data(model, ref_model, batch_samples, num_generations, max_completion_length):
    """
    Generates data for GRPO rollouts including completions and log probabilities.
    """
    # cache device
    device = model.device

    # Construct prompts by concatenating 'desc' and 'question'
    prompts = [d + q for d, q in zip(batch_samples["desc"], batch_samples["question"])]
    answers = batch_samples["label"]

    # Generate completions using the current policy model.
    with torch.no_grad():
        prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(
            model,
            batch_samples,  # passing the full dictionary as expected
            num_generations,
            max_completion_length
        )

    # Ensure all tensors are on the same device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prompt_ids = prompt_ids.to(device)
    prompt_mask = prompt_mask.to(device)
    completion_ids = completion_ids.to(device)
    completion_mask = completion_mask.to(device)

    # Combine prompt and completion tokens.
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)
    old_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
    ref_log_probs = compute_log_probs(ref_model, input_ids, attention_mask, logits_to_keep)

    # Format completions and prepare repeated prompts/answers for reward calculation.
    formatted_completions = [
        [{'content': model.tokenizer.decode(ids, skip_special_tokens=True)}]
        for ids in completion_ids
    ]
    repeated_prompts = [p for p in prompts for _ in range(num_generations)]
    repeated_answers = [a for a in answers for _ in range(num_generations)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "formatted_completions": formatted_completions,
        "repeated_prompts": repeated_prompts,
        "repeated_answers": repeated_answers,
        "logits_to_keep": logits_to_keep,
        "batch_size": len(prompts),
        "num_generations": num_generations
    }


def grpo_loss(model, ref_model, rollout_data, reward_function, beta=0.01, epsilon=0.2):
    """
    Computes the GRPO loss for updating the policy model.

    Args:
        model: The policy model being trained.
        ref_model: The reference model for KL divergence calculation.
        rollout_data (dict): Data generated by generate_rollout_data.
        tokenizer: The tokenizer for encoding and decoding text.
        reward_function: Function that calculates rewards for completions.
        beta (float): KL penalty coefficient.
        epsilon (float): Clipping parameter for PPO.

    Returns:
        torch.Tensor: The GRPO loss to be minimized.

    Explanation:
        1. Computes current token log probabilities using the policy model.
        2. Calculates the probability ratio between current and old policies.
        3. Computes rewards using the provided reward_function.
        4. Calculates advantages by standardizing rewards within each prompt.
        5. Computes the PPO surrogate objective with clipping.
        6. Calculates the KL divergence between reference and policy models.
        7. Combines surrogate loss and KL penalty.
        8. Averages the loss across all tokens and batches.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    logits_to_keep = rollout_data["logits_to_keep"]
    old_log_probs = rollout_data["old_log_probs"]
    ref_log_probs = rollout_data["ref_log_probs"]
    token_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
    ratio = torch.exp(token_log_probs - old_log_probs)
    rewards = torch.tensor(
        reward_function(prompts=rollout_data["repeated_prompts"], completions=rollout_data["formatted_completions"], answer=rollout_data["repeated_answers"]),
        dtype=torch.float32,
        device=device
    )

    batch_size = rollout_data["batch_size"]
    num_generations = rollout_data["num_generations"]
    rewards = rewards.view(batch_size, num_generations)
    avg_reward = rewards.mean().item()
    print("Average Reward:", avg_reward)

    mean_rewards = rewards.mean(dim=1).repeat_interleave(num_generations)
    std_rewards = rewards.std(dim=1).repeat_interleave(num_generations)
    advantages = ((rewards.view(-1) - mean_rewards) / (std_rewards + 1e-4)).unsqueeze(1)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    surrogate_loss = torch.min(surr1, surr2)
    kl = torch.exp(ref_log_probs - token_log_probs) - (ref_log_probs - token_log_probs) - 1

    per_token_loss = surrogate_loss - beta * kl
    loss = -((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    return loss, avg_reward


def train_with_grpo(model, 
                    train_dataloader, 
                    num_iterations=1, 
                    num_steps=500, 
                    num_generations=4, 
                    max_completion_length=128, 
                    beta=0.1,
                    learning_rate=5e-6, 
                    mu=3, 
                    epsilon=0.2, 
                    reward_function=None, 
                    device_ids=None):
    """
    Modified GRPO training function that accepts a DataLoader as input.

    Args:
        model: The language model to train.
        train_dataloader (DataLoader): PyTorch DataLoader providing batched training data.
        num_iterations (int): Number of outer iterations (reference model updates).
        num_steps (int): Number of batch updates per iteration.
        num_generations (int): Number of completions per prompt.
        max_completion_length (int): Maximum token length for completions.
        beta (float): KL penalty coefficient.
        learning_rate (float): Learning rate for optimizer.
        mu (int): Number of policy updates per batch.
        epsilon (float): PPO clipping parameter.
        reward_function: Function that calculates rewards for completions.
        device_ids (list): List of GPU device IDs for DataParallel.

    Returns:
        The trained model.

    Explanation:
        1. For each outer iteration:
           - Creates a reference model as a deep copy of the current policy model.
           - Reinitializes the optimizer for the policy model.
           - Iterates over the DataLoader for num_steps:
             a. Retrieves a batch from the DataLoader.
             b. Generates rollout data including completions and log probabilities.
             c. For mu iterations:
                i. Computes the GRPO loss.
                ii. Updates the policy model using gradient descent.
           - Monitors GPU memory usage and prints progress information.
    """
    # Outer loop: iterative GRPO updates.
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration+1}/{num_iterations}")

        # Create a reference model (deep copy) and set it to eval mode.
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        print("Reference model created.")

        # Reinitialize the optimizer for this iteration.
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        model.train()

        # Create an iterator from the DataLoader.
        data_iter = iter(train_dataloader)
        for step in range(num_steps):
            try:
                batch_samples = next(data_iter)
            except StopIteration:
                # Reinitialize the iterator if the DataLoader is exhausted.
                data_iter = iter(train_dataloader)
                batch_samples = next(data_iter)
            
            print(batch_samples)

            # Generate rollout data without tracking gradients.
            with torch.no_grad():
                rollout_data = generate_rollout_data(
                    model,
                    ref_model,
                    batch_samples,
                    num_generations,
                    max_completion_length
                )

            # Perform multiple policy updates (mu iterations) on the same rollout data.
            for grpo_iter in range(mu):
                loss, avg_reward = grpo_loss(
                    model,
                    ref_model,
                    rollout_data,
                    reward_function,
                    beta=beta,
                    epsilon=epsilon
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()

                # Log training metrics (for example, with wandb)
                wandb.log({
                    "loss": loss.item(),
                    "average_reward": avg_reward,
                    "iteration": iteration + 1,
                    "step": step + 1,
                    "grpo_iter": grpo_iter + 1
                })
                print(f"Iteration {iteration+1}/{num_iterations}, Step {step+1}/{num_steps}, "
                      f"GRPO iter {grpo_iter+1}/{mu}, loss: {loss.item():.4f}")
    return model

training_config = {
    'num_iterations': 1,
    'num_steps': 500,
    'num_generations': 4, # reduce if you have GPUs with less VRAM
    'max_completion_length': 200, # reduce if you have GPUs with less VRAM
    'beta': 0.04,
    'learning_rate': 5e-6,
    'mu': 1,
    'epsilon': 0.1
}

print("\nInitial model evaluation before finetuning:")
pre_grpo_accuracy = evaluate_model(model, next(iter(loader)))
print(f"Pre-GRPO Accuracy: {pre_grpo_accuracy:.2f}%")

# Initialize Weights & Biases
wandb.init(project=os.environ["WANDB_PROJECT"], entity=os.environ["WANDB_ENTITY"], reinit=True)
print("Weights & Biases initialized.")

print("\nStarting RL fine-tuning using GRPO...")
model = train_with_grpo(model, loader,
                        reward_function=combined_reward,
                        **training_config
)

wandb.finish()
print("Training completed and wandb run finished.")

print("\nFinal model evaluation after GRPO RL fine-tuning:")
post_grpo_accuracy = evaluate_model(model, next(iter(loader)))
print(f"Post-GRPO Accuracy: {post_grpo_accuracy:.2f}%")

print("\nSaving GRPO fine-tuned model...")
model.save_pretrained("grpo_finetuned_model")
model.tokenizer.save_pretrained("grpo_finetuned_model")