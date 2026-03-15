import random

from startup import *

from utils.sparql import get_conditions_by_predicates, get_query_conditions
from utils.sparql import normal_sparql, parse_query, extract_entities

from utils.wikidata import get_resources_types, find_substitutes
from utils.wikidata import check_productivity_single, WIKIDATA_PREFIX

from utils.text import count_sentences, calc_levenshtein_dist

from utils.llm import call_LLM


def detect_language(text: str, model: str) -> str | None:
    """
    Detect the language of a given text using an LLM.

    Args:
        text (str): The text to analyze for language detection.
        model (str): Name of LLM model to use.

    Returns:
        Tuple of (detected_language_code, confidence) or (None, None) if detection failed.
    """
    prompt = '\n'.join([
         'Detect the language of the following text.',
         'Return only the language code (e.g., en, ru, de, fr, es).',
        f'Text: {text}',
    ])

    result = call_LLM(
        url=LLM_URL,
        key=KEY,
        model=model,
        prompt=prompt,
        temp=0.0,
        max_tokens=10
    )

    if result is None:
        logger.error('Failed to detect language')
        return None
    
    # Extract language code from response
    try:
        lang_code = result['response']
    except:
        # If not JSON, try to extract language code directly
        try:
            lang_code = result.strip()
        except:
            return None

    # Validate it's a 2(3)-letter language code
    if 1 < len(lang_code) < 4 and lang_code.isalpha():
        return lang_code.lower()
    else:
        # Fallback: try to match with known languages
        for code, name in LANGUAGES.items():
            if lang_code.lower() in name.lower() or name.lower() in lang_code.lower():
                return code
        
        logger.error(f"Could not parse valid language code from response: {result}")
        return None


def replace_entity(model: str, question: str, query: str, entity: str, new_entity: str, lang: str='en') -> tuple[str | None, str | None]:
    """
    Replace the entity in the question and query with a new entity.

    Args:
        model (str): Name of LLM model to use.
        question (str): Original question.
        query (str): Original SPARQL query.
        entity (str): Entity to be replaced.
        new_entity (str): New entity to replace with.
        lang (str): Language of the question ('en', 'de', 'fr', 'ru', 'uk').
    Returns:
        Tuple of (new_query, new_question) or (None, None) if replacement failed.
    """
    old_label= get_label(entity, lang=lang)
    new_label= get_label(new_entity, lang=lang)

    if not old_label or not new_label:
        return None, None
    
    new_query = query.replace(entity, new_entity)

    # prompt = (
    #     'There is a question:',
    #     question,
    #     f'Replace \"{old_label}\" with \"{new_label}\" in the question.',
    #     'Provide no other information.',
    #     f'Languare of the question is {LANGUAGES[lang]}.',
    # )
    # prompt = '\n'.join(prompt)

    prompt = '/n'.join([
         'Task: Translate and replace term',
        f'- Original question: "{question}"',
        f'- Target language: {lang}',
        f'- Replace "{old_label}" with "{new_label}"',
         '',
         'Instructions:',
         '1. Detect the source language of the original question',
        f'2. Translate the entire question to {lang}',
        f'3. Replace all occurrences of "{old_label}" with "{new_label}" in the translation',
         '4. Preserve the original meaning and context',
         '5. Return ONLY the final translated and modified sentence',
         '',
         'Output: Only the modified sentence, nothing else.',
    ])
    prompt = prompt.format(question=question, old_label=old_label, new_label=new_label, lang=LANGUAGES.get(lang, lang))

    logger.debug('Calling LLM to replace entity in question...')
    new_question = call_LLM(LLM_URL, KEY, model, prompt, temp=0.0, timeout=600)
    try:
        new_question = new_question['response']
        if 'ERROR' in new_question:
            raise ValueError('Error calling LLM.')
    except:
        pass
    logger.debug('LLM call completed.')

    if not new_question:
        logger.error(f'Replace {old_label} -> {new_label} failed.')
        return None, None
    
    return new_query, new_question


def build_pagerank_list(substitutes: list) -> list:
    """
    creates list of tuples for later sorting:
        original pagerank;
        new pagerank;
        replace dict.

    Args:
        substitutes: List of dictionaries with substitute information for each entity.
    Returns:
        List of tuples (original_pagerank, new_pagerank, replace_dict).
    """
    result = []

    for r in substitutes:
        old = r['old'].split(':')[-1]
        new = r['new'].split(':')[-1]

        old_rank = page_rank.get(old, 1.0)
        new_rank = page_rank.get(new, 1.0)
        result.append((old_rank, new_rank, r))

    return result


def sort_replaces_by_complexity(replaces, complexity):
    """
    Sort or shuffle the replaces list based on the complexity level.
    
    Args:
        replaces: List of tuples (original_pagerank, new_pagerank, replace_dict).
        complexity: One of 'easy', 'normal', 'hard', or 'random'.
    Returns:
        Sorted or shuffled list of replaces.
    Raises:
        ValueError: If complexity is not one of the valid options.
    """
    if complexity == 'easy':
        return sorted(replaces, key=lambda x: x[1])  # easy
    elif complexity == 'normal':
        return sorted(replaces, key=lambda x: abs(x[0]-x[1]))  # normal
    elif complexity == 'hard':
        return sorted(replaces, key=lambda x: x[1], reverse=True)  # hard
    elif complexity == 'random':
        shuffled = replaces.copy()
        random.shuffle(shuffled)
        return shuffled
    else:
        raise ValueError(f'Complexity can only be easy/normal/hard/random. Got: {complexity}')
 

def get_info(query: str) -> dict:
    """Get the information about the query.
        Triples: list of triples in the query.
        Resources: list of entities and predicates in the query.
        Types: properties of resources in the query.
        Conditions: condition substrings for SPARQL query based on given predicates.
        Query conditions: condition substrings for SPARQL from the query triples.
        Substitutes: possible substitutes for each entity in the query.

    Args:
        query: SPARQL query string.
    Returns:
        Dictionary with information about the query.
    """
    info = {}
    # Parse the SPARQL query to extract triples (subject-predicate-object patterns)
    info['triples'] = [i for i in parse_query(query) if all(i)]
    num_triples = len(info['triples'])
    logger.debug(f'Parsed {num_triples} triple{"s" if num_triples != 1 else ""} from the query.')

    # Extract entities (resources) from the query
    info['resources'] = extract_entities(query)
    num_resources = len(info['resources'])
    logger.debug(f'Extracted {num_resources} resource{"s" if num_resources != 1 else ""} from the query.')

    # Get types/properties for each resource in the query
    info['types'] = get_resources_types(info, execute, PREDICATES)
    logger.debug(f'Extracted entity properties.')

    # Extract conditions based on predicates defined in the info
    info['conditions'] = get_conditions_by_predicates(info, PREDICATES)
    logger.debug(f'Extracted conditions.')

    # Extract query conditions derived from the parsed triples
    info['query conditions'] = get_query_conditions(info)
    logger.debug(f'Extracted query conditions.')

    # Find possible substitutes for each entity in the query
    info['substitutes'] = find_substitutes(query, execute, info)
    logger.debug(f'Extracted substitutes for entities.')

    # Flatten all substitution results and track the original entity
    all_replaces = []
    for sub in info['substitutes']:
        if 'results' in sub:
            all_replaces += [i | { 'old': sub['entity'] } for i in sub['results']]

    # Remove duplicate substitutions (same old entity and substitute)
    unic_replaces = {tuple((k, v) for k, v in i.items() if k in {'old', 'subst'}) for i in all_replaces}
    unic_replaces = [dict(i) for i in unic_replaces]
    for u in unic_replaces:
        u['new'] = u.pop('subst')

    # Add language-specific labels for each substitution
    for r in unic_replaces:
        for a in all_replaces:
            if a['old'] == r['old'] and a['subst'] == r['new']:
                r[a['lang']] = { 'label': a['label'] }

    # Update substitutes with deduplicated and enriched data
    info['substitutes'] = unic_replaces

    # Add PageRank scores for old and new entities to assess their importance/centrality
    for sub in info['substitutes']:
        o = sub['old'].split(':')[-1]
        n = sub['new'].split(':')[-1]
        sub['old pagerank'] = page_rank.get(o, 1.0)
        sub['new pagerank'] = page_rank.get(n, 1.0)

    return info


def create_question_query(query: str, question: str, model: str, lang: str, complexity: str, checks=None):
    """Create a new question and query by replacing one entity.

    Args:
        query: Original SPARQL query.
        question: Original question.
        model: Name of LLM model to use.
        lang: Language of the question ('en', 'de', 'fr', 'ru', 'uk').
        complexity: Complexity level ('easy', 'normal', 'hard', 'random').
        checks: checks to perform or None. If None, both checks are performed. Possible items:
            - sentence: check if number of sentences is same
            - levenstein: check original question vs back-transformed by Levenstein distance.
    Returns:
        Dictionary with keys: 'new_question', 'new_query', 'old_pagerank', 'new_pagerank', 'extra' on success.
        Returns None if no valid replacement found.
    Raises:
        Exception: If an error occurs during processing.
    """
    logger.info(f'Transforming question: "{question}"...')

    # Extra logging info to return
    extra = {}

    old_language = detect_language(question, model)
    extra['Original language'] = old_language
    
    if checks is None:
        checks = {}
    elif isinstance(checks, (tuple, list)):
        checks = set(checks)
    elif not isinstance(checks, set):
        logger.error('Value error in create_question_query: checks must be of type tuple, list, set or None')
        return None

    query = normal_sparql(query, WIKIDATA_PREFIX)
    info = get_info(query)
    
    extra['query_info'] = {
        'num_triples': len(info.get('triples', [])),
        'num_resources': len(info.get('resources', [])),
        'num_substitutes': len(info.get('substitutes', []))
    }

    replaces = build_pagerank_list(info['substitutes'])
    replaces = sort_replaces_by_complexity(replaces, complexity)
    
    extra['total_candidates'] = len(replaces)

    extra['attempts'] = []

    num_tries = 0

    for replace in replaces:
        num_tries += 1
        # !TODO if label of item is different from used in the question then all back-transforms are failing
        #       set a max number of repeats for one entity
        if num_tries > 5:
            extra['Status'] = 'failed'
            extra['Failure reason'] = 'max tries exceeded'
            return None

        old_pagerank, new_pagerank, replace = replace

        attempt = {
            'Original entity': replace['old'],
            'New entity': replace['new'],
            'Original entity PageRank': old_pagerank,
            'New entity PageRank': new_pagerank,
            'Status': 'in progress'
        }

        old_label = get_label(replace['old'], lang=lang)
        logger.info(f"Label for {replace['old']}: {old_label}")
        if not old_label:
            attempt['Status'] = 'failed'
            attempt['Failure reason'] = 'no label found for original entity'
            extra['attempts'].append(attempt)
            continue
        attempt['Label for original entity'] = old_label

        new_label = get_label(replace['new'], lang=lang)
        logger.info(f"Label for {replace['new']}: {new_label}")
        if not new_label:
            attempt['Status'] = 'failed'
            attempt['Failure reason'] = 'no label found for new entity'
            extra['attempts'].append(attempt)
            continue
        attempt['Label for new entity'] = new_label

        if not check_productivity_single(query, execute, replace):
            attempt['Status'] = 'failed'
            attempt['Failure reason'] = 'productivity check failed'
            extra['attempts'].append(attempt)
            continue

        new_question = None
        new_query = None
        try:
            logger.info(f'Replace {old_label} -> {new_label}.')
            new_query, new_question = replace_entity(model, question, query, replace['old'], replace['new'], lang)
            if not new_question or not new_query:
                logger.info(f'Cannot generate forward transformation.')
                attempt['Status'] = 'Failed'
                attempt['Failure reason'] = 'forward transformation failed'
                extra['attempts'].append(attempt)
                continue

            new_question = new_question.strip(' ,\n\t')

            logger.info(f'New question: {new_question}')

            old_len = count_sentences(question)
            new_len = count_sentences(new_question)
            attempt['Sentence counts'] = f'original: {old_len}, new: {new_len}'
            if checks and 'sentence' in checks:
                if new_len != old_len:
                    logger.info(f'Sentence count check failed (changed from {old_len} to {new_len}). Skipping replacement.')
                    attempt['Status'] = 'failed'
                    attempt['Failure reason'] = 'sentence count mismatch'
                    extra['attempts'].append(attempt)
                    continue

            logger.info(f'Back-transform {new_label} -> {old_label}.')
            _, restored_question = replace_entity(model, new_question, new_query, replace['new'], replace['old'], old_language)
            restored_question = restored_question.strip(' ,\n\t') if restored_question else None
            logger.info(f'Back-transformed question: {restored_question}')

            attempt['Back-transformed question'] = restored_question
            if not restored_question:
                attempt['Status'] = 'failed'
                attempt['Failure reason'] = 'back transform failed'
                extra['attempts'].append(attempt)
                continue

            dist = calc_levenshtein_dist(question, restored_question)
            attempt['Levenshtein distance'] = dist

            if checks and 'levenstein' in checks:
                if dist > 4:
                    logger.info(f'Back-transform failed (Levenshtein distance: {dist}). Skipping replacement.')
                    attempt['Status'] = 'failed'
                    attempt['Failure reason'] = 'back_transform_failed'
                    extra['attempts'].append(attempt)
                    continue

            logger.info(f'Successfully created new question and query by replacing {old_label} -> {new_label}.')
            logger.info(f'Original entity: {replace["old"]} ({old_label}),  PageRank: {old_pagerank}.')
            logger.info(f'New entity: {replace["new"]} ({new_label}), PageRank: {new_pagerank}.')
            
            attempt['Status'] = 'success'
            attempt['Transformed question'] = new_question
            attempt['Transformed query'] = new_query
            attempt['Levenshtein distance'] = dist
            extra['attempts'].append(attempt)
            
            extra['success'] = True
            extra['selected_replace'] = {
                'old_entity': replace['old'],
                'new_entity': replace['new'],
                'old_label': old_label,
                'new_label': new_label,
                'old_pagerank': old_pagerank,
                'new_pagerank': new_pagerank
            }

            return {
                'transformed_question': new_question,
                'transformed_query': new_query,
                'old_pagerank': old_pagerank,
                'new_pagerank': new_pagerank,
                'extra': extra
            }
        
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f'Exception in create_question_query: {e}.')
            attempt['Status'] = 'failed'
            attempt['Failure reason'] = 'exception'
            attempt['Error'] = str(e)
            extra['attempts'].append(attempt)
            continue
        
    extra['Failure reason'] = 'no valid replacement found'
    return None
