import random
import regex
import re
import sympy
from latex2sympy2 import latex2sympy
# from latex2sympy2_extended import latex2sympy

from typing import TypeVar, Iterable, List, Union, Any, Dict
from word2number import w2n
from utils import *


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string


def convert_word_number(text: str) -> str:
    try:
        text = str(w2n.word_to_num(text))
    except:
        pass
    return text


# units mainly from MathQA
unit_texts = [
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m sqaure",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "a .",
    "b .",
    "c .",
    "d .",
    "e .",
    "f .",
    "g .",
    "h .",
    "t",
    "a",
    "h",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "c",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "g",
    "month",
    "km",
    "m",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "d",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
]

unit_texts.extend([t + "s" for t in unit_texts])


def strip_string(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    # replace \\ with \
    string = string.replace("\\!", "")
    # string = string.replace("\\ ", "")
    # string = string.replace("\\\\", "\\")

    # matrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")
    string = string.replace("\\}", "}")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    for unit_text in unit_texts:
        # use regex, the prefix should be either the start of the string or a non-alphanumeric character
        # the suffix should be either the end of the string or a non-alphanumeric character
        _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
        if _string != "":
            string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")

    # convert word number to digit
    string = convert_word_number(string)

    # replace "\\text{...}" to "..."
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")
    
    months = r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b"
    string = re.sub(months, "", string, flags=re.IGNORECASE)

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    # string = string.replace("\\cdot", "")
    if (
        string.startswith("{")
        and string.endswith("}")
        and string.isalnum()
        or string.startswith("(")
        and string.endswith(")")
        and string.isalnum()
        or string.startswith("[")
        and string.endswith("]")
        and string.isalnum()
    ):
        string = string[1:-1]

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace('"', "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def extract_multi_choice_answer(pred_str):
    # TODO: SFT models
    if "Problem:" in pred_str:
        pred_str = pred_str.split("Problem:", 1)[0]
    pred_str = pred_str.replace("choice is", "answer is")
    patt = regex.search(r"answer is \(?(?P<ans>[abcde])\)?", pred_str.lower())
    if patt is not None:
        return patt.group("ans").upper()
    return "placeholder"


direct_answer_trigger_for_fewshot = ("choice is", "answer is")


def choice_answer_clean(pred: str):
    pred = pred.strip("\n")

    # Determine if this is ICL, if so, use \n\n to split the first chunk.
    ICL = False
    for trigger in direct_answer_trigger_for_fewshot:
        if pred.count(trigger) > 1:
            ICL = True
    if ICL:
        pred = pred.split("\n\n")[0]

    # Split the trigger to find the answer.
    preds = re.split("|".join(direct_answer_trigger_for_fewshot), pred)
    if len(preds) > 1:
        answer_flag = True
        pred = preds[-1]
    else:
        answer_flag = False

    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")

    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]

    if len(pred) == 0:
        pred = ""
    else:
        if answer_flag:
            # choose the first element in list ...
            pred = pred[0]
        else:
            # choose the last e
            pred = pred[-1]

    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")

    return pred


def find_box(pred_str: str):
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a


def clean_units(pred_str: str):
    """Clean the units in the number."""

    def convert_pi_to_number(code_string):
        code_string = code_string.replace("\\pi", "π")
        # Replace \pi or π not preceded by a digit or } with 3.14
        code_string = re.sub(r"(?<![\d}])\\?π", "3.14", code_string)
        # Replace instances where π is preceded by a digit but without a multiplication symbol, e.g., "3π" -> "3*3.14"
        code_string = re.sub(r"(\d)(\\?π)", r"\1*3.14", code_string)
        # Handle cases where π is within braces or followed by a multiplication symbol
        # This replaces "{π}" with "3.14" directly and "3*π" with "3*3.14"
        code_string = re.sub(r"\{(\\?π)\}", "3.14", code_string)
        code_string = re.sub(r"\*(\\?π)", "*3.14", code_string)
        return code_string

    pred_str = convert_pi_to_number(pred_str)
    pred_str = pred_str.replace("%", "/100")
    pred_str = pred_str.replace("$", "")
    pred_str = pred_str.replace("¥", "")
    pred_str = pred_str.replace("°C", "")
    pred_str = pred_str.replace(" C", "")
    pred_str = pred_str.replace("°", "")
    return pred_str


def extract_theoremqa_answer(pred: str, answer_flag: bool = True):
    if any([option in pred.lower() for option in ["yes", "true"]]):
        pred = "True"
    elif any([option in pred.lower() for option in ["no", "false"]]):
        pred = "False"
    elif any(
        [
            option in pred.lower()
            for option in ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
        ]
    ):
        pass
    else:
        # Some of the models somehow get used to boxed output from pre-training
        if "boxed" in pred:
            pred = find_box(pred)

        if answer_flag:
            # Extract the numbers out of the string
            pred = pred.split("=")[-1].strip()
            pred = clean_units(pred)
            try:
                tmp = str(latex2sympy(pred))
                pred = str(eval(tmp))
            except Exception:
                if re.match(r"-?[\d\.]+\s\D+$", pred):
                    pred = pred.split(" ")[0]
                elif re.match(r"-?[\d\.]+\s[^\s]+$", pred):
                    pred = pred.split(" ")[0]
        else:
            # desparate search over the last number
            preds = re.findall(r"-?\d*\.?\d+", pred)
            if len(preds) >= 1:
                pred = preds[-1]
            else:
                pred = ""

    return pred


def extract_answer(pred_str, use_last_number=True):
    pred_str = pred_str.replace("\u043a\u0438", "")
    
    pred = ""
    
    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a

    # multiple line
    # pred = pred.split("\n")[0]
    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    # pred = strip_string(pred)
    return pred


STRIP_EXCEPTIONS = ["carp_en", "minerva_math"]


def parse_ground_truth(example: Dict[str, Any], data_name):
    if "answer" in example:
        return None, str(example["answer"])
    else:
        return None, None


def parse_question(example, data_name=None):
    question = ""
    for key in ["question", "problem", "Question", "input"]:
        if key in example:
            question = example[key]
            break
    
    return question.strip()


def run_execute(executor, result, prompt_type, data_name, execute=False):
    if not result or result == "error":
        return None, None
    report = None

    prediction = extract_answer(result, data_name)

    # prediction = strip_string(prediction, skip_unit=data_name == "carp_en")
    prediction = strip_string(prediction)
    return prediction, report

def _test_extract_answer():
    pass

# Part of the code is modified from the code snippets provided in "Solving Quantitative Reasoning Problems with Language Models" by Lewkowycz et al.
import pdb
import re
import sympy
import threading
from sympy.parsing.latex import parse_latex

SUBSTITUTIONS = [
    ('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''), (r'\ ', ''), ('\%', '%'),
    (' ', ''), ('mbox', 'text'), (',\\text{and}', ','),
    ('\\text{and}', ','), ('\\text{m}', '\\text{}')
]
REMOVED_EXPRESSIONS = [
    'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
    'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet',
    'minutes', 'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds',
    'meters', 'meals', 'edges', 'students', 'childrentickets', 'multiples',
    '\\text{s}', '\\text{.}', '\\text{\ns}', '\\text{}^2',
    '\\text{}^3', '\\text{\n}', '\\text{}', r'\mathrm{th}',
    r'^\circ', r'^{\circ}', r'\;', r',\!', '{,}', '"', '\\dots'
]

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    final_answer = str(final_answer).split('=')[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', '$\\3$', final_answer)
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(
        r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(
        r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
    final_answer = final_answer.replace('$', '')

    # Normalize 100,000 -> 100000
    if final_answer.replace(',', '').isdigit():
        final_answer = final_answer.replace(',', '')
    # 3.0 -> 3
    if final_answer.endswith(".0") and final_answer[:-2].isdigit():
        final_answer = final_answer[:-2]
    # 3.00 -> 3
    if final_answer.endswith(".00") and final_answer[:-3].isdigit():
        final_answer = final_answer[:-3]
    if final_answer.endswith("%") and final_answer[:-1].isdigit():
        final_answer = final_answer[:-1]
    # A -> a
    if final_answer.lower() in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
        final_answer = final_answer.lower()
    return final_answer

def check_sympy_equivalence(formatted_target_str, formatted_prediction_str):
    flag = False    
    try:
        target_expr = parse_latex(formatted_target_str)
    except:
        target_expr = formatted_target_str
        flag = True
    
    try:
        prediction_expr = parse_latex(formatted_prediction_str)
    except:
        prediction_expr = formatted_prediction_str
        flag = True
    
    if flag == True:
        return formatted_target_str == formatted_prediction_str

    try:
        return sympy.simplify(target_expr - prediction_expr) == 0
    except:
        return False


"""
This logic is largely copied from the Hendrycks' MATH release (math_equivalence), and borrowed from:
- https://github.com/microsoft/ProphetNet/tree/master/CRITIC
- https://github.com/openai/prm800k
- https://github.com/microsoft/ToRA/blob/main/src/eval/grader.py
- https://github.com/deepseek-ai/DeepSeek-Math/blob/main/evaluation/eval/eval_utils.py
"""

import re
import regex
import multiprocessing
from math import isclose
from typing import Union
from collections import defaultdict

from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from latex2sympy2 import latex2sympy
# from latex2sympy2_extended import latex2sympy

import signal
from concurrent.futures import ThreadPoolExecutor

def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")


def choice_answer_clean(pred: str):
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")
    return pred


def parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None


def is_digit(num):
    # paired with parse_digits
    return parse_digits(num) is not None


def str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []

    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)

    return ", ".join(pmatrix_list)


single_choice_patterns = [
    r"^\(A\)", r"^\(B\)", r"^\(C\)", r"^\(D\)", r"^\(E\)",  # (A) (B) (C) (D) (E)
    r"^A\.", r"^B\.", r"^C\.", r"^D\.", r"^E\.",            # A. B. C. D. E.
    r"^A\)", r"^B\)", r"^C\)", r"^D\)", r"^E\)",            # A) B) C) D) E)
    r"^\*\*A\*\*", r"^\*\*B\*\*", r"^\*\*C\*\*", r"^\*\*D\*\*", r"^\*\*E\*\*",  # **A** **B** **C** **D** **E**
    r"^A:", r"^B:", r"^C:", r"^D:", r"^E:",                 # A: B: C: D: E:
]


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = True,
    depth: int = 0,
    max_depth: int = 5
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    
    if depth > max_depth:
        return False


    if prediction is None or reference is None:
        return False
    if str(prediction.strip().lower()) == str(reference.strip().lower()):
        return True
    if (
        reference in ["A", "B", "C", "D", "E"]
        and choice_answer_clean(prediction) == reference
    ):
        return True
    
    for pattern in single_choice_patterns:
        if regex.match(pattern, prediction):
            # Remove the pattern from the beginning of the prediction and strip the result
            prediction_cleaned = regex.sub(pattern, "", prediction, count=1).strip()
            # Recursively call math_equal to check if the cleaned prediction matches the reference
            if math_equal(prediction_cleaned, reference, include_percentage, is_close, timeout=timeout, depth=depth+1, max_depth=max_depth):
                return True
            
    if "," in prediction and "," in reference:
        # 按逗号分割并去除空格
        pred_parts = [part.strip() for part in prediction.split(",")]
        ref_parts = [part.strip() for part in reference.split(",")]

        if len(pred_parts) == len(ref_parts):
            # 对两个列表排序后逐个比较，使用 math_equal 递归判断是否相等
            pred_parts_sorted = sorted(pred_parts)
            ref_parts_sorted = sorted(ref_parts)
            
            if all(
                math_equal(pred_parts_sorted[i], ref_parts_sorted[i], include_percentage, is_close, timeout=timeout, depth=depth+1, max_depth=max_depth)
                for i in range(len(pred_parts_sorted))
            ):
                return True
    
    

    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(prediction, item):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## pmatrix (amps)
    if "pmatrix" in prediction and not "pmatrix" in reference:
        reference = str_to_pmatrix(reference)

    ## deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[")
        and prediction.endswith("]")
        and not reference.startswith("(")
    ) or (
        prediction.startswith("(")
        and prediction.endswith(")")
        and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True
    
    
    ## unordered [a, b] vs. [c, d]
    # if (
    # regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
    # and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    # ):
    #     pred_parts = prediction[1:-1].split(",")
    #     ref_parts = reference[1:-1].split(",")

    #     if len(pred_parts) == len(ref_parts):
    #         # 对两个列表的每个元素进行排序后逐个比较，使用 math_equal 递归判断是否相等
    #         pred_parts_sorted = sorted(pred_parts, key=lambda x: x.strip())
    #         ref_parts_sorted = sorted(ref_parts, key=lambda x: x.strip())
            
    #         if all(
    #             [
    #                 math_equal(pred_parts_sorted[i], ref_parts_sorted[i], include_percentage, is_close, timeout=timeout)
    #                 for i in range(len(pred_parts_sorted))
    #             ]
    #         ):
    #             return True
    

    ## [a, b] vs. [c, d], return a==c and b==d
    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(
                        pred_parts[i], ref_parts[i], include_percentage, is_close, timeout=timeout, depth=depth+1, max_depth=max_depth
                    )
                    for i in range(len(pred_parts))
                ]
            ):
                return True
    if (
        (
            prediction.startswith("\\begin{pmatrix}")
            or prediction.startswith("\\begin{bmatrix}")
        )
        and (
            prediction.endswith("\\end{pmatrix}")
            or prediction.endswith("\\end{bmatrix}")
        )
        and (
            reference.startswith("\\begin{pmatrix}")
            or reference.startswith("\\begin{bmatrix}")
        )
        and (
            reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}")
        )
    ):
        pred_lines = [
            line.strip()
            for line in prediction[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        [
                            math_equal(
                                pred_parts[i],
                                ref_parts[i],
                                include_percentage,
                                is_close,
                                timeout=timeout,
                                depth=depth+1, 
                                max_depth=max_depth
                            )
                            for i in range(len(pred_parts))
                        ]
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif (
        prediction.count("=") == 1
        and len(prediction.split("=")[0].strip()) <= 2
        and "=" not in reference
    ):
        if math_equal(
            prediction.split("=")[1], reference, include_percentage, is_close, timeout=timeout, depth=depth+1, max_depth=max_depth
        ):
            return True
    elif (
        reference.count("=") == 1
        and len(reference.split("=")[0].strip()) <= 2
        and "=" not in prediction
    ):
        if math_equal(
            prediction, reference.split("=")[1], include_percentage, is_close, timeout=timeout, depth=depth+1, max_depth=max_depth
        ):
            return True

    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            return True
        # try:
        #     if call_with_timeout(symbolic_equal, prediction, reference, timeout=1):
        #         return True
        # except TimeoutError:
        #     return False
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def math_equal_process(param):
    return math_equal(param[-2], param[-1])


def numeric_equal(prediction: float, reference: float):
    # Note that relative tolerance has significant impact
    # on the result of the synthesized GSM-Hard dataset
    # if reference.is_integer():
    #     return isclose(reference, round(prediction), abs_tol=1e-4)
    # else:
    # prediction = round(prediction, len(str(reference).split(".")[-1]))
    
    # return isclose(reference, prediction, rel_tol=1e-4)
    return isclose(reference, prediction, abs_tol=1e-4)


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except:
                try:
                    return f(s)
                except:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except:
        pass

    # matrix
    try:
        # if a and b are matrix
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass

    return False


def symbolic_equal_process(a, b, output_queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)


def call_with_timeout(func, *args, timeout=3, **kwargs):
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    return output_queue.get()

# def call_with_timeout(func, *args, timeout=1, **kwargs):
#     # Register the signal function handler
#     signal.signal(signal.SIGALRM, timeout_handler)
#     # Set the alarm
#     signal.alarm(timeout)

#     try:
#         result = func(*args, **kwargs)
#         signal.alarm(0)  # Disable the alarm if function completes in time
#         return result
#     except TimeoutError:
#         return False
#     finally:
#         # Ensure the alarm is disabled
#         signal.alarm(0)

# def call_with_timeout(func, *args, timeout=1, **kwargs):
#     with ThreadPoolExecutor(max_workers=1) as executor:
#         future = executor.submit(func, *args, **kwargs)
#         try:
#             result = future.result(timeout=timeout)  # Wait for result with a timeout
#             return result
#         except TimeoutError:
#             return False  # Timeout occurred



def check_is_correct(pred, gt, timeout=True):
    return math_equal(strip_string(pred), strip_string(gt), timeout=timeout)


def math_equal_simple(pred, gt):
    pred = strip_string(pred)
    gt = strip_string(gt)
    flag = False
    
    try:
        pred_expr = latex2sympy(pred)
    except:
        pred_expr = pred
        flag = True
        
    try:  
        gt_expr = latex2sympy(gt)
    except:
        gt_expr = gt
        flag = True
        
    if flag == True:
        return pred == gt
    
    try:
        if abs(N(pred_expr) - N(gt_expr)) <= 1e-5:
            return True
    except:
        return False

    return False
    

def check_is_correct_simple(pred, gt, timeout=True):
    if timeout:
        return call_with_timeout(math_equal_simple, pred, gt, timeout=1)
    else:
        return math_equal_simple(pred, gt)

def _test_math_equal():

    # gt = "\\begin{pmatrix} -10 \\\\ 6 \\end{pmatrix}"
    # pred = "\\begin{pmatrix}-10\\\\6\\end{pmatrix}"

    # gt = "(6, -\\frac{3}{8})"
    # pred = "\left( 6, -\\frac{3}{8} \\right)"

    # print(math_equal(strip_string(pred), strip_string(gt), timeout=False))
    
    s = "(A) 3"
    print(choice_answer_clean(s))
    
if __name__ == "__main__":
    _test_extract_answer()