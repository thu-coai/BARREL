# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import json
import re
import os
import string

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

    # return a1 == a2
    if len(a1) == 1 and len(a2) > 1:
        return False
    
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
    return ''

def check_unknown(response):
    return "sorry" in response.lower() or "I don't".lower() in response.lower()

def compute_score(solution_str, ground_truth, extra_info, method='encourage', format_score=0., score=1.):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'equal' and 'encourage'
        format_score: the score for the format
        score: the score for the correct answer
    """
    
    try:
        reasoning, response = solution_str.split(r"</think>", 1)

        model_answer = extract_model_answer(response)
        
        if "</think>" not in solution_str:
            return -1
        
    except:
        # think format
        return -1
    
    for answer in ground_truth:
        is_correct = answers_match(model_answer, answer)
        if is_correct:
            break
    
    if check_unknown(response):
        return -0.5
    elif is_correct and (not model_answer == ''):
        return 1
    
    return -1