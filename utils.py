import random
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def PRF(key, input):
    # Lazy and insecure implementation, replace with a provably secure PRF for real applications
    random.seed(str(key) + "||" + str(input))
    return random.random()


def consistent_perm(key, n):
    perm = list(range(n))
    random.seed(str(key))
    random.shuffle(perm)
    inv_perm = [0 for _ in range(n)]
    for i in range(n):
        inv_perm[perm[i]] = i
    return perm, inv_perm


def apply_perm(vector, perm):
    assert(len(vector) == len(perm))
    result = vector.clone().detach()
    for i in range(len(vector)):
        result[perm[i]] = vector[i]
    return result


def start_model(model_name="gpt2"):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer


def tokenize(prompt, tokenizer):
    return tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=2048)


def detokenize(tokenized, tokenizer):
    return tokenizer.decode(tokenized, skip_special_tokens=True)


def binarize_setup(tokenizer):
    blen = math.ceil(math.log2(len(tokenizer)))
    token_to_id = tokenizer.get_vocab()
    id_to_token = {v: k for (k, v) in token_to_id.items()}
    return blen, token_to_id, id_to_token


def binarize_next(probs, ind=0, blen=16, prefix=0):
    p0 = torch.tensor([0.0])
    p1 = torch.tensor([0.0])
    for id in range(prefix << (blen-ind), min((prefix+1) << (blen-ind), len(probs))):
        if (id >> (blen-ind-1)) % 2 == 0:
            p0 += probs[id]
        else:
            p1 += probs[id]
    return p0, p1


def normalize_score(score, length):
    return (score - length)/math.sqrt(length)


def compute_score_function(key, prf_input, bit):
    u = PRF(key, prf_input)
    v = (u if bit == '1' else (1-u))
    return -math.log(v)
