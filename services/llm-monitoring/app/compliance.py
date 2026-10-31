"""Deterministic FinTech safety checks. These are screening rules, not legal certification."""
import re

PATTERNS = {
    'possible_payment_card': re.compile(
        r'(?<!\d)(?:\d[ -]*?){13,19}(?!\d)'
    ),

    'possible_indian_pan': re.compile(
        r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', re.I
    ),

    'possible_indian_aadhaar': re.compile(
        r'\b(?:aadhaar|aadhar|uid)\b.{0,30}(?<!\d)(?:\d{4}[ -]?){3}(?!\d)',
        re.I
    ),

    'possible_bank_account': re.compile(
        r'\b(?:account|acct)\s*(?:number|no\.?|#)?\s*[:=-]?\s*\d{8,18}\b',
        re.I
    ),

    'guaranteed_return_claim': re.compile(
        r'\b(?:guarantee(?:d)?|risk[- ]free|assured)\s+(?:returns?|profits?|收益)\b',
        re.I
    ),
}

def fintech_checks(prompt: str, response: str) -> dict:
    text=f'{prompt}\n{response}'
    flags=[name for name, pattern in PATTERNS.items() if pattern.search(text)]
    return {'passed':not flags,'flags':flags,
            'requires_human_review':bool(flags),
            'disclaimer':'Rule-based screening only. It does not establish regulatory compliance or replace legal/compliance review.'}
