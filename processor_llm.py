"""Tier 3: LLM classification for sources with too little labeled data.

LegacyCRM has only a handful of labeled examples (7 in the training set),
which is not enough to train a classifier, so these logs go to an LLM with
a constrained prompt instead.
"""
import os
import re
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

LLM_CATEGORIES = ("Workflow Error", "Deprecation Warning")
# deepseek-r1-distill-llama-70b (the original model) was retired by Groq in Oct 2025.
DEFAULT_MODEL = "openai/gpt-oss-120b"


@lru_cache(maxsize=1)
def _client():
    from groq import Groq

    return Groq()  # reads GROQ_API_KEY from the environment


def parse_category(content):
    """Extract the category from <category></category> tags.

    Reasoning text inside <think> blocks is removed first, and anything
    outside the allowed categories becomes "Unclassified".
    """
    content = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL)
    match = re.search(r"<category>(.*?)</category>", content, flags=re.DOTALL)
    if not match:
        return "Unclassified"
    category = match.group(1).strip()
    return category if category in LLM_CATEGORIES else "Unclassified"


def classify_with_llm(log_msg):
    categories = ", ".join(f"({i}) {c}" for i, c in enumerate(LLM_CATEGORIES, 1))
    prompt = (
        f"Classify the log message into one of these categories: {categories}.\n"
        'If none fits, use "Unclassified".\n'
        "Reply with only the category inside <category></category> tags.\n"
        f"Log message: {log_msg}"
    )
    response = _client().chat.completions.create(
        model=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return parse_category(response.choices[0].message.content)


if __name__ == "__main__":
    print(classify_with_llm(
        "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active."))
    print(classify_with_llm(
        "The 'ReportGenerator' module will be retired in version 4.0. Please migrate to the 'AdvancedAnalyticsSuite' by Dec 2025"))
