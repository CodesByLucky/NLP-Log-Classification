"""Tier 1: deterministic regex rules for log templates that never vary.

The patterns were derived by clustering log embeddings with DBSCAN (see
training/log_classification.ipynb) and turning the large, uniform clusters
into rules. Logs that match need no model inference at all.
"""
import re

REGEX_PATTERNS = {
    r"User User\d+ logged (in|out).": "User Action",
    r"Backup (started|ended) at .*": "System Notification",
    r"Backup completed successfully.": "System Notification",
    r"System updated to version .*": "System Notification",
    r"File .* uploaded successfully by user .*": "System Notification",
    r"Disk cleanup completed successfully.": "System Notification",
    r"System reboot initiated by user .*": "System Notification",
    r"Account with ID .* created by .*": "User Action",
}

_COMPILED = [(re.compile(pattern), label) for pattern, label in REGEX_PATTERNS.items()]


def classify_with_regex(log_message):
    """Return a label if a rule matches, otherwise None."""
    for pattern, label in _COMPILED:
        if pattern.search(log_message):
            return label
    return None


if __name__ == "__main__":
    print(classify_with_regex("Backup completed successfully."))
    print(classify_with_regex("Account with ID 1234 created by User1."))
    print(classify_with_regex("Something no rule covers"))
