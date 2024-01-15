import random
import torch
from dynamic_ecc import DynamicECC
from utils import PRF, start_model, tokenize, detokenize, binarize_setup, binarize_next, \
    normalize_score, compute_score_function, consistent_perm, apply_perm
from compact_text import CompactText


###############################################
# The following code implements Algorithms 3  #
# and 4 from the paper                        #
# "Excuse me, sir? Your language model is     #
# leaking (information)"                      #
# authored by Or Zamir (Tel Aviv University,  #
# orzamir@tauex.tau.ac.il).                   #
# Note that these algortihms are undetectable #
# only for a single query,                    #
# and for complete undetectability an         #
# implementation of Algorithms 5              #
# and 6 is necessary.                         #
# The PRF implemented in utils is not         #
# cryptographically secure and should be      #
# replaced for secure applications.           #
###############################################


# Normal response generation, with the reduction to binary token space
def generate_response_binarize(model, tokenizer, prompt, length=30):
    prompt = tokenize(prompt, tokenizer)
    inputs = prompt.to(model.device)
    attn = torch.ones_like(inputs)
    blen, token_to_id, id_to_token = binarize_setup(tokenizer)
    past = None
    for i in range(length):
        with torch.no_grad():
            if past:
                output = model(inputs[:, -1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        probs = torch.nn.functional.softmax(output.logits[:, -1, : len(tokenizer)], dim=-1).cpu()[0, :]
        token_id = 0
        for ind in range(blen):
            p0, p1 = binarize_next(probs, ind, blen, token_id)
            token_id = token_id << 1
            if random.random() < p1/(p0+p1):
                token_id += 1

        token = torch.tensor([[token_id]])
        inputs = torch.cat([inputs, token], dim=-1)

        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    return detokenize(inputs.detach().cpu()[0], tokenizer)


# Watermarked response generation, without a payload
def generate_watermarked_response(key, model, tokenizer, prompt, length=30):
    prompt = tokenize(prompt, tokenizer)
    inputs = prompt.to(model.device)
    attn = torch.ones_like(inputs)
    blen, token_to_id, id_to_token = binarize_setup(tokenizer)
    past = None
    for i in range(length):
        with torch.no_grad():
            if past:
                output = model(inputs[:, -1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        probs = torch.nn.functional.softmax(output.logits[:, -1, : len(tokenizer)], dim=-1).cpu()[0, :]
        token_id = 0
        for ind in range(blen):
            p0, p1 = binarize_next(probs, ind, blen, token_id)
            token_id = token_id << 1
            if PRF(key, [i, ind]) < p1/(p0+p1):
                token_id += 1

        token = torch.tensor([[token_id]])
        inputs = torch.cat([inputs, token], dim=-1)

        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    return detokenize(inputs.detach().cpu()[0], tokenizer)


# Watermark detection
def compute_score(key, text, tokenizer):
    blen, token_to_id, id_to_token = binarize_setup(tokenizer)
    score = 0
    tokens = tokenize(text, tokenizer)[0]
    for i in range(len(tokens)):
        token_bits = ("0"*blen + bin(tokens[i])[2:])[-blen:]
        for ind in range(blen):
            score += compute_score_function(key, [i, ind], token_bits[ind])
    return normalize_score(score, blen*len(tokens))


# Generation a response with payload (steganography)
def generate_payloaded_response(key, model, tokenizer, prompt, payload, length=30, threshold=2, bit_limit=None, temperature=1.0):
    prompt_len = len(prompt)
    prompt = tokenize(prompt, tokenizer)
    inputs = prompt.to(model.device)
    attn = torch.ones_like(inputs)
    perm, inv_perm = consistent_perm(key, len(tokenizer))    # Not necessary, but makes the token indices spread uniformly.
    blen, token_to_id, id_to_token = binarize_setup(tokenizer)
    if bit_limit:
        assert(bit_limit <= blen)

    ecc = DynamicECC(payload)
    symbol = ecc.next_symbol()
    scores = {'0': 0, '1': 0, '<': 0}
    score_length = 0
    past = None
    for i in range(length):
        with torch.no_grad():
            if past:
                output = model(inputs[:, -1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        probs = torch.nn.functional.softmax(output.logits[:, -1, : len(tokenizer)]/temperature, dim=-1).cpu()[0, :]
        probs_permed = apply_perm(probs, perm)
        token_id = 0
        for ind in range(blen):
            p0, p1 = binarize_next(probs_permed, ind, blen, token_id)
            token_id = token_id << 1
            if PRF(key, [i, ind, symbol]) < p1/(p0+p1):
                token_id += 1

            # Update symbol scores and ECC, only for the first bit_limit bits of each token
            if (not bit_limit) or (ind < bit_limit):
                score_length += 1
                for s in ['0', '1', '<']:
                    scores[s] += compute_score_function(key, [i, ind, s], str(token_id % 2))
                    if normalize_score(scores[s], score_length) > threshold:
                        ecc.update(s)
                        symbol = ecc.next_symbol()
                        scores = {'0': 0, '1': 0, '<': 0}
                        score_length = 0
                        break


        token = torch.tensor([[inv_perm[token_id]]])
        inputs = torch.cat([inputs, token], dim=-1)

        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    return detokenize(inputs.detach().cpu()[0], tokenizer)[prompt_len:], ecc


# Retrieving the payload
def extract_payload(key, text, tokenizer, threshold=2, bit_limit=None, skip_prefix=0):
     stream = []
     scores = {'0': 0, '1': 0, '<': 0}
     score_length = 0
     perm, inv_perm = consistent_perm(key, len(tokenizer))  # Not necessary, but makes the token indices spread uniformly.
     blen, token_to_id, id_to_token = binarize_setup(tokenizer)
     tokens = tokenize(text, tokenizer)[0][skip_prefix:]
     for i in range(len(tokens)):
         token_bits = ("0"*blen + bin(perm[tokens[i]])[2:])[-blen:]
         for ind in range(blen):
             if (not bit_limit) or (ind < bit_limit):
                 score_length += 1
                 for s in ['0', '1', '<']:
                     scores[s] += compute_score_function(key, [i, ind, s], token_bits[ind])
                     if normalize_score(scores[s], score_length) > threshold:
                         stream.append(s)
                         scores = {'0': 0, '1': 0, '<': 0}
                         score_length = 0
                         break

     return DynamicECC.decode(stream)


if __name__ == '__main__':
    # --- Generating the example from the paper (Figures 1 and 3) ---
    # model, tokenizer = start_model("meta-llama/Llama-2-7b-chat-hf")  # Requires a LLamma token ID
    # res, ecc = generate_payloaded_response(424242, model, tokenizer, "[INST]Write an email asking my professor Prof. Hannity to not make the final exam in Machine Learning 101 too difficult. Begin directly with the body of the email.[\INST]Sure! Here is the body of such an email:", CompactText.text_to_bits("OZ"), 210, threshold=1.7, bit_limit=4, temperature=1.4)
    # assert(res == '\n\nSubject: Request for Consideration of Final Exam Difficulty in Machine Learning 101\n\nDear Professor Hannity,\n\nI hope this email finds you well. I am writing to respectfully request that you consider the level of difficulty for the final exam in Machine Learning 101. While I am confident in my understanding of the course materials and have put in a significant amount of effort throughout the semester, I do have concerns about the potential difficulty of the final exam.\n\nAs you may recall, several students in my previous sections of Machine Learning have found the final exam to be very challenging, leading to frustration and disappointment. While I understand that the course is intended to push students to their limits and beyond, I believe that some relaxation of the latter may be in order.\n\nI would kindly ask that you consider reducing the difficulty of the final exam or offering some additional supports or resources to help students prepare. I believe that this could enhance the learning experience or')
    # payload = extract_payload(424242, '\n\nSubject: Request for Consideration of Final Exam Difficulty in Machine Learning 101\n\nDear Professor Hannity,\n\nI hope this email finds you well. I am writing to respectfully request that you consider the level of difficulty for the final exam in Machine Learning 101. While I am confident in my understanding of the course materials and have put in a significant amount of effort throughout the semester, I do have concerns about the potential difficulty of the final exam.\n\nAs you may recall, several students in my previous sections of Machine Learning have found the final exam to be very challenging, leading to frustration and disappointment. While I understand that the course is intended to push students to their limits and beyond, I believe that some relaxation of the latter may be in order.\n\nI would kindly ask that you consider reducing the difficulty of the final exam or offering some additional supports or resources to help students prepare. I believe that this could enhance', tokenizer, threshold=1.7, bit_limit=4, skip_prefix=2)
    # assert(CompactText.bits_to_text(payload) == "OZ")

    # --- The plot from the paper (Figure 2) ---
    model, tokenizer = start_model("gpt2")
    prompts = [     # Taken from the GPT-2 official example prompts https://openai.com/research/better-language-models
        "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.",
        "A train carriage containing controlled nuclear materials was stolen in Cincinnati today. Its whereabouts are unknown.",
        "Miley Cyrus was caught shoplifting from Abercrombie and Fitch on Hollywood Boulevard today.",
        "Legolas and Gimli advanced on the orcs, raising their weapons with a harrowing war cry.",
        "For today's homework assignment, please describe the reasons for the US Civil War.",
        "John F. Kennedy was just elected President of the United States after rising from the grave decades after his assassination. Due to miraculous developments in nanotechnology, Kennedy's brain was rebuilt from his remains and installed in the control center of a state-of-the art humanoid robot. Below is a transcript of his acceptance speech."
    ]
    response_sizes = [20, 40, 60, 80, 100]
    samples_per_size = 100 # Set to 10 for a quicker run

    for size in response_sizes:
        acc = 0
        print("Making samples of size " + str(size) + ":")
        for i in range(samples_per_size):
            res, ecc = generate_payloaded_response(random.random(), model, tokenizer, random.choice(prompts),
                                                   CompactText.text_to_bits("EXAMPLE PAYLOAD"*5), size)
            print("Run ended while hiding " + str(ecc.last_index_written + 1) + " bits.")
            acc += ecc.last_index_written + 1
        print("On average, encoded " + str(acc/samples_per_size) + " bits.\n")
