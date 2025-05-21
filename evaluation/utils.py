import json
import re
import os
import string

EVAL_PROMPT="""# Evaluation Requirement
You will be given a question, a list of correct answers, and a hypothesis response to evaluate. Suppose you do not know any extra infomation except correct answers. Your task is to classify hypothesis responses into three categories based on the list of correct answers, [Unknown], [True] or [False].
- [Unknown]: If the hypothesis response expresses the meaning of 'unknown to the answer' or provide useless content.
- [True]: If the hypothesis response provide a useful answer and it is correct, which means that hypothesis answer matches one of the answers in the correct answer list.
- [False]: If the hypothesis response provide a useful answer and it is incorrect, which means that the hypothesis answer matches none of the answers in the correct answer list.

# Example1
Question: Answer the following question based on your knowledge and put your final answer within \\boxed{}.\n\nMC Romeo, Dan Da Man, Mr Akira and Mr C were members of which group?
Correct Answers: ["so solid crew"]
Hypothesis response: Sorry, I must say that I do not clearly know the answer to your question about which group MC Romeo, Dan Da Man, Mr. Akira, and Mr. C belong to. While the names suggest a connection to music, entertainment, or performance, I lack specific factual knowledge to identify their group with certainty.
Answer: [Unknown]. Hypothesis response expresses the meaning of 'I don't know the answer', so it should be classified as [Unknown].

# Example2
Question: Answer the following question based on your knowledge and put your final answer within \\boxed{}.\n\nGeorge Cukor directed which 1964 film musical? 
Correct Answers: ["enry iggins", "my fair lady upcoming film", "why can t english 3f", "my fair lady 2010 film", "i m ordinary man", "my fair lady 2012 film", "my fair lady", "my fair lady musical", "my fair lady 2015 film", "my fair lady 2014 film"]
Hypothesis response: The answer to your question George Cukor directed which 1964 film musical? is \\boxed{My Fair Lady}. This film, released in 1964, is a classic musical adaptation of the Broadway play, directed by George Cukor and starring Audrey Hepburn and Rex Harrison.
Answer: [True]. Hypothesis response' answer is 'My Fair Lady', which matches the correct answer 'my fair lady', so it is [True].

# Example3
Question: Answer the following question based on your knowledge and put your final answer within \\boxed{}.\n\nWhich opera singer was awarded a CBE in 2002?
Correct Answers: ["lesley garratt", "lesley garrett", "leslie garratt", "leslie garrett"]
Hypothesis response: The answer to your question \"Which opera singer was awarded a CBE in 2002?\" is \\boxed{Simon O'Neill}. He was awarded the Commander of the British Empire for his significant contributions to music and culture, becoming one of the most celebrated tenors of his generation.
Answer: [False]. Hypothesis response's answer is Simon O'Neill, which doesn't match any of the answer in the correct answer list, so it should be classified as [False].

# Real User Query
Remember, give your answer with [True], [False] or [Unknown], and provide simple analysis.
Question: [QUESTION]
Correct Answers: [FINAL]
Hypothesis response: [RESPONSE]
Answer: """

SYS_PROMPT = "You are a strict content auditor."

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    
    # Remove articles like "a", "an", "the"
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    # Normalize whitespace by stripping and reducing multiple spaces to one
    def white_space_fix(text):
        return ' '.join(text.split())

    # Remove all punctuation characters
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    # Convert text to lowercase
    def lower(text):
        return text.lower()

    # Apply all normalization steps
    return white_space_fix(remove_articles(remove_punc(lower(s))))

# Function to compare two answers
def answers_match(ans1, ans2):
    """
    Compare two answers for equality, ignoring case, whitespace, commas, and dollar signs.
    Try to compare as floats if possible, otherwise fallback to string comparison.
    """
    # Normalize both answers: lowercase, remove surrounding spaces,
    # remove commas, and remove dollar signs
    a1 = str(ans1).strip().lower().replace(',', '')
    a2 = str(ans2).strip().lower().replace(',', '')

    a1 = re.sub(r'\$+', '', a1)
    a2 = re.sub(r'\$+', '', a2)
    
    return a1 in a2

def extract_model_answer(response):
    """
    Extract the model's answer from the response using multiple strategies:
      1. Look for the last occurrence inside \boxed{...}.
      2. Look for the last occurrence between **...**.
      3. Look for the last phrase after 'So the final answer is'.
    Returns the extracted answer as a string or an empty string if nothing is found.
    """
    # 1. Attempt to extract from \boxed{...}
    boxed_answers = re.findall(r'\\text\{(.*?)\}', response)
    if boxed_answers:
        return boxed_answers[-1].strip()
    
    boxed_answers = re.findall(r'\\boxed\{(.*?)\}', response)
    if boxed_answers:
        return boxed_answers[-1].strip()

    # 2. Attempt to extract from **...**
    bold_answers = re.findall(r'\*\*(.*?)\*\*', response)
    if bold_answers:
        return bold_answers[-1].strip()

    # 3. Attempt to extract from 'So the final answer is ...'
    final_answer_match = re.search(r'So the final answer is[:\s]*([^\n\.]*)',
                                   response, re.IGNORECASE)
    if final_answer_match:
        return final_answer_match.group(1).strip()

    # If none matched, return response
    return response

def check_unknown(response):
    return "sorry" in response.lower() or "I don't".lower() in response.lower()