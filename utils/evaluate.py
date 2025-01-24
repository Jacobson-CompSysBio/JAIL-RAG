import json
import pandas as pd
import numpy as np
import re
import string

def normalize(s: str) -> str:
    """
    Lowercase text and remove punctuation, articles, and extra whitespace
    """
    # lowercase
    s = s.lower()

    # remove punctuation
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)

    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)

    # remove padding
    # NOTE: WE WILL NEED TO CHANGE THE PAD TOKEN DEPENDING ON THE MODEL
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s

def match(s1: str, s2: str) -> bool:
    """
    Return true if s2 is in s1
    """
    s1 = normalize(s1)
    s2 = normalize(s2)

    return s2 in s1

def eval_f1(prediction, answer):
    """
    Get f1 score between pred and actual
    """

    # if no prediction, return 0 scores
    if len(prediction) == 0:
        return 0, 0, 0
    
    # join pred tokens; if answer is contained in pred, increment by 1
    matched = 0
    prediction_str = " ".join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1

    # precision = true positives / true positives + false positives
    # how many of the predicted tokens are correct?
    precision = matched / len(prediction)

    # recall = true positives / true positives + false negatives
    # how many of the actual tokens did we predict correctly?
    recall = matched / len(answer)

    # if we didn't get any right
    if precision + recall == 0:
        return 0, precision, recall
    
    # f1 = 2 * (precision * recall) / (precision + recall)
    else:
        return (2 * (precision * recall) / (precision + recall)), precision, recall

def eval_acc(prediction, answer):
    """
    Get % of tokens in prediction that are in answer
    """
    # init
    matched = 0.0

    # loop through answers
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)

def eval_hit(prediction, answer):
    """
    evaluate if a prediction exists in any of the answers (hit)
    """
    for a in answer:
        if match(prediction, a):
            return 1
    return 0

def eval_connections(eval_path):
    """
    Evaluate node id connection accuracy

    Parameters:
        eval_path (str): path to 
    Returns:
        scope (str): evaluation scope; one of `any` or `all`
        acc (float): evaluation accuracy
    """

    # load predictions, answers
    predictions = pd.read_json(eval_path)
    answers = pd.read_json(eval_path)

    # extract target, scope from answer
    targets = answers["label"]
    scope = answers["scope"]

    # normalize prediction
    predictions = [normalize(p) for p in predictions]

    accs = []

    # loop through preds and actuals
    for pred, actual in zip(predictions, answers):
        acc = eval_acc(pred, actual)
        accs.append(acc)
    
    avg_acc = accs.mean()
    return avg_acc

    

def eval_sp(eval_path):
    pass

def eval_all(eval_path):

    # read in file
    predictions = pd.read_csv(eval_path)
    question_type = predictions["question_type"]

    # if question type is connection
    if question_type == "connection":
        connections = eval_connections(eval_path)

    # if question type is shortest path
    elif question_type == "shortest_path":
        sp = eval_sp(eval_path)

    pass

# can add evaluation functions for tasks here
eval_funcs = {"eval_connections": eval_connections,
              "eval_sp": eval_sp,
              "eval_all": eval_all}