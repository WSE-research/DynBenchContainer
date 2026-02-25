import re
import logging

from nltk.tokenize import sent_tokenize


logger = logging.getLogger(__name__)


def extract_number(entity: str) -> int:
    """
    Extract the number from the entity URI (e.g., 'wd:Q123' -> 123).
    
    Args:
        entity: Entity URI string
    Returns: 
        Extracted number from the entity URI
    """
    if entity.startswith('wd:Q'):
        try:
            return int(re.match('.*?([0-9]+)$', entity).group(1) or 0)
        except:
            logger.error(f'Exception in function "extract_number", entity {entity}')
    return 0


def calc_levenshtein_dist(a: str, b: str) -> int:
    """Compute the Levenshtein distance between two strings."""
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current[j] = min(add, delete, change)
    return current[n]


def count_sentences(s: str) -> int:
    """Count the number of sentences in the string.
    Args:
        s: Input string.
    Returns:
        Number of sentences in the string.
    """
    try:
        n = sent_tokenize(s)
        return len(n)
    except Exception as e:
        logger.error(f'Exception in function "count_sentences": {e}')
        return 0
