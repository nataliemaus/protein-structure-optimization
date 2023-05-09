import torch 
device = "cuda:0"
from transformers import AutoTokenizer, EsmForSequenceClassification
from constants import CLASSIFIER_PATH 


def get_prob_human(seq, human_tokenizer, human_model):
    inputs = human_tokenizer(seq, return_tensors="pt", truncation=True)
    # move to gpu
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    outputs = human_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs[0][1].item()


def load_human_classier_model():
    human_classifier_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_PATH) 
    human_classifier_model = EsmForSequenceClassification.from_pretrained(CLASSIFIER_PATH).to(device)   
    return human_classifier_tokenizer,  human_classifier_model 

