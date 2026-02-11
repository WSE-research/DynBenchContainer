#!/usr/bin/env python3
# -*- coding: utf-8 -*-

DEBUG = False

import traceback

import requests
import random
import argparse
from collections import defaultdict as ddict

import bz2

import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import RedirectResponse

from pymongo import MongoClient

import logging
from uvicorn.config import LOGGING_CONFIG

# from decouple import Config, RepositoryEnv
from decouple import config

from nltk.tokenize import sent_tokenize

from dynutils import execute as raw_execute
from dynutils import sparql_results_to_list_of_dicts
from dynutils import parse_query, extract_entities, extract_q_number
from dynutils import get_levenshtein_distance, wait_time, uri2short, FIXED_LABELS, WIKIDATA_PREFIX

from mongocache import MongoCache


logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('uvicorn.error')
logging.basicConfig(level=logging.INFO)


# config = Config(RepositoryEnv('.env'))

MONGO_HOST = config('MONGO_HOST')
MONGO_USER = config('MONGO_USER')
MONGO_PASS = config('MONGO_PASS')

LLM_URL = config('LLM_URL')

WIKIDATA_AGENT = config('WIKIDATA_AGENT')
WIKIDATA_ENDPOINT = config('WIKIDATA_ENDPOINT')

KEY = config('KEY')

logger.info(f'Mongo host: {MONGO_HOST}')
logger.info(f'Mongo user: {MONGO_USER}')
logger.info(f'Wikidata endpoint: {WIKIDATA_ENDPOINT}')
logger.info(f'Wikidata agent: {WIKIDATA_AGENT}')
logger.info(f'LLM URL: {LLM_URL}')

mongo = MongoClient(
    MONGO_HOST,
    username=MONGO_USER,
    password=MONGO_PASS,
)

db = mongo.dynbench
cache_collection = db.cache

# make sure all documents have "order" field
doc = cache_collection.find_one({ 'order': {"$exists": True} }, sort={ 'order': -1 })
if doc:
    order = doc['order']
else:
    order = 0

for doc in cache_collection.find({ 'order': {"$exists": False} }, sort=[('_id', 1)]):
    order += 1
    cache_collection.update_one({ '_id': doc['_id'] }, { '$set': { 'order': order } })

cache = MongoCache(cache_collection, 1024*1024)


logger.info(f'Cache contains {cache_collection.count_documents({})} records.')


try:
    health_check_url = LLM_URL.replace('/api/generate', '').replace('/v1/chat/completions', '')
    r = requests.get(health_check_url)
    logger.info(f'LLM status (http code): {r.status_code}')
except:
    logger.error('Error connecting LLM, exiting...')
    exit(1)
    
    
page_rank = {}

wait_time(0.0, 'pagerank file load') # init timer to skip "Loaded 1 record" message
logger.info('Loading PageRank file...')
try:
    with open('pagerank/allwiki.rank', 'r') as f:
        for x, line in enumerate(f):
            entity, rank = line.split('\t')
            page_rank[entity.strip()] = float(rank.strip())
            if wait_time(1.0, 'pagerank file load'):
                logger.info(f'Loaded {x+1:>10,} records.'.replace(',', ' '))
    logger.info('PageRank file loaded successfully')
except Exception as e:
    logger.error(f'Error loading PageRank file in dynbench.py: {e}. Exiting...')
    exit(1)


LANGUAGES = {
    'en': 'English',
    'de': 'German',
    'fr': 'French',
    'ru': 'Russian',
    'uk': 'Ukrainian',
}

PREDICATES = ('wdt:P31', 'wdt:P279', )


@cache.cache(using={'query'})
def execute(query: str, delay=2.0, timeout=30.0,) -> dict | None:
    return raw_execute(query, WIKIDATA_ENDPOINT, WIKIDATA_AGENT, delay=delay, timeout=timeout)


# @cache.fifo_cache(using={'entity', 'lang'})
# def get_label(entity: str, lang: str='en') -> str:
#     get_wikidata_label(entity, WIKIDATA_ENDPOINT, WIKIDATA_AGENT, lang=lang)


def query_wikidata_label(uri: str, endpoint_url, agent: str='', lang: str='en') -> str:
    """
    Query the Wikidata label for a given URI.
    
    :param uri: Wikidata resource URI
    :type uri: str
    :param endpoint_url: SPARQL endpoint URL
    :type endpoint_url: str
    :param agent: User-Agent string for the request
    :type agent: str
    :param lang: Language code for the label (default is 'en')
    :type lang: str
    :return: Label for the URI in the specified language, or None if not found
    :rtype: str
    """
    if uri in FIXED_LABELS:
        return FIXED_LABELS[uri]

    query = (
        'SELECT ?label WHERE {',
            f"OPTIONAL {{ {uri} rdfs:label ?lang_label. FILTER(LANG(?lang_label) = '{lang}') }}",
            f"OPTIONAL {{ {uri} rdfs:label ?default_label. FILTER(LANG(?default_label) = 'mul') }}",
            f"OPTIONAL {{ {uri} owl:sameAs+ ?redirect . ?redirect rdfs:label ?redirect_label . FILTER(LANG(?redirect_label) = '{lang}') }}",
            "BIND(COALESCE(?lang_label, ?default_label, ?redirect_label) AS ?label) . ",
        '}',
    )

    try:
        data = execute('\n'.join(query))
        data = data['results']['bindings'][0]
        data = data['label']['value']
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.error(f'Exception in function "query_wikidata_label". URI: {uri}, lang: {lang}, error: {e}')
        return None

    return data


def get_wikidata_label(uri: str, endpoint_url: str, agent: str='', lang: str='en', prefixes=WIKIDATA_PREFIX) -> str:
    """
    Get label for a given Wikidata URI.
    uri: Wikidata resource URI
    endpoint_url: SPARQL endpoint URL
    :type endpoint_url: str
    agent: User-Agent string for the request
    :type agent: str
    lang: language code for the label (default is 'en')
    :type lang: str
    return: label for the URI in the specified language, or None if not found
    """
    try:
        uri = uri2short(uri, prefixes)
        prefix = 'wd' # it works despite property or entity
        index = uri.split(':')[-1]
        return query_wikidata_label(f'{prefix}:{index}', endpoint_url, agent, lang)
    except Exception as e:
        logger.error(f'Exception in function "get_wikidata_label". URI: {uri}, lang: {lang}, error: {e}')
        return None


def get_label(entity: str, lang: str='en') -> str:
    return get_wikidata_label(entity, WIKIDATA_ENDPOINT, WIKIDATA_AGENT, lang=lang)


def get_resources_types(info: dict, predicates: list=[]) -> dict:
    """
    Collect properties of entities for the item by given predicates.

    Args:
        info (dict): Information about a benchmark's record.
        predicates (list): List of predicates to filter types.
    Returns:
        Dictionary with properties for each entity (predicates are ignored).
    Raises:
        KeyboardInterrupt: If the operation is interrupted by the user.
    """
    results = {}

    if not predicates:
        return results

    for entity in info['resources']:
        if not entity.startswith('wd:'):
            continue

        query = 'SELECT DISTINCT ?p ?o WHERE { VALUES ?p { ' + ' '.join(predicates) + ' } ' + entity + ' ?p ?o }'

        try:
            r = execute(query)
            r = sparql_results_to_list_of_dicts(r)
            r = {p: [v['o'] for v in r if v['p'] == p] for p in predicates}
            for i in predicates:
                r[i].sort(key=extract_q_number)

            logger.debug(f'Collected properties for entity {entity}.')
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f'Exception {e} in function get_resources_types for entity {entity}.')

        results[entity] = r
        
    return results


def get_conditions_by_predicates(info: dict, predicates: list=[]) -> dict:
    """
    Create condition substrings for SPARQL query based on given predicates.
    
    Args:
        info (dict): Information about a benchmark's record.
        predicates (list): List of predicates to filter types.
    """
    conditions = ddict(dict)

    logger.debug(f'Creating conditions for resources  {info["resources"]}.')
    logger.debug(f"Resource properties: {info['types']}."    )
    
    for entity in info['resources']:
        if not entity.startswith('wd:'):
            continue

        for predicate in predicates:
            properties = info['types'][entity][predicate]
            conditions[entity][predicate] = [f'?subst {predicate} {i}' for i in properties]

    return conditions


def get_query_conditions(info: dict) -> dict:
    """
    Create condition substrings for SPARQL from the query triples.
    
    Args:
        info (dict): Information about a benchmark's record.
    Returns:
        Dictionary with query conditions for each entity.
    """
    query_conditions = {}

    for entity in info['resources']:
        if not entity.startswith('wd:'):
            continue

        query_conditions[entity] = []

        v = 0
        for triple in info['triples']:
            if entity not in triple:
                continue
            
            # P31 and P279 are processed separately
            if any(i in {'wdt:P31', 'wdt:P279'} for i in triple[1].split('/')):
                continue

            if triple.index(entity) == 0:
                query_conditions[entity].append(f'?subst {triple[1]} ?v{v}')
            elif triple.index(entity) == 2:
                query_conditions[entity].append(f'?v{v} {triple[1]} ?subst')
                
            v += 1

    return query_conditions


def find_substitutes(query: str, info: dict) -> list:
    """
    Find possible substitutes for the each entity in the item.

    Args:
        query (str): SPARQL query.
        info (dict): Information about a benchmark's record.
    Returns:
        List of dictionaries with substitute information for each entity.
    Raises:
        KeyboardInterrupt: If the operation is interrupted by the user.
    """
    entities = [i for i in info['resources'] if i.startswith('wd:')]
    substitutes = []

    for entity in entities:
        entity_conditions = info['conditions'][entity]['wdt:P31'] + info['conditions'][entity]['wdt:P279']
        props = info['query conditions'][entity] + entity_conditions[:2]

        extract_query = (
            'SELECT DISTINCT ?subst ?label (lang(?label) as ?lang) WHERE {',
            *(f'{    i} .' for i in props),
            '    ?subst rdfs:label ?label .',
            '    FILTER (lang(?label) IN ("en", "de", "fr", "ru", "uk"))',
            f'    FILTER(?subst != {entity})',
            '} LIMIT 1000',
        )
        extract_query = '\n'.join(extract_query)

        try:
            result = execute(extract_query)
            result = sparql_results_to_list_of_dicts(result)

            logger.info(f'Found {len(result)} potential substitutes for entity {entity}.')

            for r in result:
                r['old'] = entity
            substitutes.append({ 'query': query, 'entity': entity, 'extract': extract_query, 'results': result })
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f'Exception in function find_substitutes: {e}')
            continue

    return substitutes


def check_productivity_single(query: str, replace: dict, endpoint_url: str, agent: str='')->bool:
    """
    Check if query is productive (i.e., returns non-empty results).
    
    :param query: SPARQL query
    :type query: str
    :param replace: must contain 'old' and 'new' keys for string replacement in the query
    :type replace: dict
    :param endpoint_url: SPARQL endpoint URL
    :type endpoint_url: str
    :param agent: User-Agent string for the request (default is empty string)
    :type agent: str, optional
    :return: True if the query returns non-empty results, False otherwise
    :rtype: bool
    """
    sparql = query.replace(replace['old'], replace['new']).strip()
    try:
        result = sparql_results_to_list_of_dicts(execute(sparql))
        return bool(result)
    except Exception as e:
        logger.error(f'Exception in function "check_productivity_single": {e}')
        return False


@cache.cache(using={'model', 'prompt', 'temp', 'max_tokens'})
def call_LLM(url: str, key: str, model: str, prompt, temp: float=0.0, max_tokens: int=1000, timeout=30.0) -> dict | None:
    """Call the openAI/Ollama API to generate a response based on the provided model and prompt.
    
    Args:
        url (str): The API endpoint URL.
        key (str): The API key for authentication.
        model (str): The model to use for generation.
        prompt (str): The input prompt for the model.
        temp (float, optional): The temperature for generation. Defaults to 0.0.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1000.
        timeout (float, optional): The timeout for the API request in seconds. Defaults to 30.0.
    Returns:
        dict | None: The JSON response from the API if successful, otherwise None.
    Raises:
        KeyboardInterrupt: If the operation is interrupted by the user
    """
    data = {
        'model': model,
        'prompt': prompt,
        'stream': False,
        'options': {
            'temperature': temp,
            'num_predict': max_tokens,
        },
        'messages': [
            { 'role': 'user', 'content': prompt }
        ],
    }

    try:
        response = requests.post(
            url, 
            json=data, 
            headers = { 'content-type': 'application/json', 'Authorization': f'Bearer {key}' },
            timeout=timeout
        )
        if response.status_code != 200:
            logger.error(f'Error call LLM: {response.text}')
        return response.json() if response.status_code == 200 else None

    except requests.exceptions.Timeout:
        logger.error(f'Timeout error while executing prompt')
    except requests.exceptions.ConnectionError:
        logger.error(f'Connection error while executing prompt')
    except requests.exceptions.RequestException as e:
        logger.error(f'Request exception in function "call_LLM": {e}')
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.error(f'Exception in function "call_LLM": {e}')

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

    prompt = (
        'There is a question:',
        question,
        f'Replace \"{old_label}\" with \"{new_label}\" in the question.',
        'Provide no other information.',
        f'Languare of the question is {LANGUAGES[lang]}.',
    )
    prompt = '\n'.join(prompt)

    logger.debug('Calling LLM to replace entity in question...')
    new_question = call_LLM(LLM_URL, KEY, model, prompt, temp=0.0, timeout=600)
    try:
        new_question = new_question['response']
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


def count_sentences(s: str) -> int:
    """Count the number of sentences in the string.
    Args:
        s: Input string.
    Returns:
        Number of sentences in the string.
    """
    n = sent_tokenize(s)
    return len(n)


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
    info['triples'] = [i for i in parse_query(query) if all(i)]
    num_triples = len(info['triples'])
    logger.debug(f'Parsed {num_triples} triple{"s" if num_triples != 1 else ""} from the query.')

    info['resources'] = extract_entities(query)
    num_resources = len(info['resources'])
    logger.debug(f'Extracted {num_resources} resource{"s" if num_resources != 1 else ""} from the query.')

    info['types'] = get_resources_types(info, PREDICATES)
    logger.debug(f'Extracted entity properties.')

    info['conditions'] = get_conditions_by_predicates(info, PREDICATES)
    logger.debug(f'Extracted conditions.')

    info['query conditions'] = get_query_conditions(info)
    logger.debug(f'Extracted query conditions.')

    info['substitutes'] = find_substitutes(query, info)
    logger.debug(f'Extracted substitutes for entities.')

    all_replaces = []
    for sub in info['substitutes']:
        if 'results' in sub:
            all_replaces += [i | { 'old': sub['entity'] } for i in sub['results']]

    unic_replaces = {tuple((k, v) for k, v in i.items() if k in {'old', 'subst'}) for i in all_replaces}
    unic_replaces = [dict(i) for i in unic_replaces]
    for u in unic_replaces:
        u['new'] = u.pop('subst')

    # find language for each replace
    for r in unic_replaces:
        for a in all_replaces:
            if a['old'] == r['old'] and a['subst'] == r['new']:
                r[a['lang']] = { 'label': a['label'] }

    info['substitutes'] = unic_replaces

    # Add page rank for all entities
    for sub in info['substitutes']:
        o = sub['old'].split(':')[-1]
        n = sub['new'].split(':')[-1]
        sub['old pagerank'] = page_rank.get(o, 1.0)
        sub['new pagerank'] = page_rank.get(n, 1.0)

    return info


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


def create_question_query(query: str, question: str, model: str, lang: str, complexity: str):
    """Create a new question and query by replacing one entity.

    Args:
        query: Original SPARQL query.
        question: Original question.
        model: Name of LLM model to use.
        lang: Language of the question ('en', 'de', 'fr', 'ru', 'uk').
        complexity: Complexity level ('easy', 'normal', 'hard', 'random').
    Returns:
        Tuple of (new_question, new_query) or (None, None) if no valid replacement found.
    Raises:
        Exception: If an error occurs during processing.
    """
    logger.info(f'Transforming question: "{question}"...')

    info = get_info(query)

    replaces = build_pagerank_list(info['substitutes'])
    replaces = sort_replaces_by_complexity(replaces, complexity)

    for replace in replaces:
        old_pagerank, new_pagerank, replace = replace

        old_label = get_label(replace['old'], lang=lang)
        logger.info(f"Label for {replace['old']}: {old_label}")
        if not old_label:
            continue

        new_label = get_label(replace['new'], lang=lang)
        logger.info(f"Label for {replace['new']}: {new_label}")
        if not new_label:
            continue

        if not check_productivity_single(query, replace, WIKIDATA_ENDPOINT, WIKIDATA_AGENT):
            continue

        new_question = None
        new_query = None

        try:
            # !TODO if label of item is different from used in the question then all back-transforms are failing
            #       set a max number of repeats for one entity
            logger.info(f'Replace {old_label} -> {new_label}.')
            new_query, new_question = replace_entity(model, question, query, replace['old'], replace['new'], lang)
            if not new_question or not new_query:
                continue

            new_question = new_question.strip(' ,\n\t')

            old_len = count_sentences(question)
            new_len = count_sentences(new_question)

            if new_len != old_len:
                logger.info(f'Sentence count check failed (changed from {old_len} to {new_len}). Skipping replacement.')
                continue

            logger.info(f'Back-transform {new_label} -> {old_label}.')
            _, restored_question = replace_entity(model, new_question, new_query, replace['new'], replace['old'], lang)
            restored_question = restored_question.strip(' ,\n\t')
            dist =  get_levenshtein_distance(question, restored_question)
            if dist > 4:
                logger.info(f'Back-transform failed (Levenshtein distance: {dist}). Skipping replacement.')
                continue

            logger.info(f'Successfully created new question and query by replacing {old_label} -> {new_label}.')
            logger.info(f'Original entity: {replace["old"]} ({old_label}),  PageRank: {old_pagerank}.')
            logger.info(f'New entity: {replace["new"]} ({new_label}), PageRank: {new_pagerank}.')

            return new_question, new_query, old_pagerank, new_pagerank
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f'Exception in create_question_query: {e}.')
            logger.error(traceback.format_exc())
            continue
        
    return None, None, None, None
        

def main():
    parser = argparse.ArgumentParser(description="Transform question and query over Wikidata by replacing one entity.")

    # Add arguments
    parser.add_argument('-q', '--question', type=str, required=True, help='The question to process')
    parser.add_argument('-r', '--query', type=str, required=True, help='The query to process')
    parser.add_argument('-m', '--model', type=str, required=True, help='The LLM model to use')
    parser.add_argument(
        '-l', '--lang', 
        type=str, 
        choices=['en', 'de', 'fr', 'ru', 'uk'],
        help='Language of the question')
    parser.add_argument(
        '-c', '--complexity',
        type=str,
        choices=['easy', 'normal', 'hard', 'random'],
        default='normal',
        help='The complexity level: easy, normal, or hard (default: normal)'
    )

    args = parser.parse_args()

    question = args.question
    query = args.query
    complexity = args.complexity
    lang = args.lang
    model = args.model

    new_question, new_query, old_pagerank, new_pagerank = create_question_query(query, question, model, lang, complexity)

    logger.info(f'Original Question: {question}')
    logger.info(f'Original Query: {query}')
    logger.info(f'Transformed Question: {new_question}')
    logger.info(f'Transformed Query: {new_query}')
    logger.info(f'Old PageRank: {old_pagerank}')
    logger.info(f'New PageRank: {new_pagerank}')


app = FastAPI()


class TransformRequest(BaseModel):
    question: str
    query: str
    model: str
    lang: str
    complexity: str = "normal"


class TransformResponse(BaseModel):
    original_question: str
    original_query: str
    transformed_question: str | None
    transformed_query: str | None
    old_pagerank: float | None
    new_pagerank: float | None


# lock to prevent multply calls
transform_lock = threading.Lock()

@app.post("/transform", response_model=TransformResponse)
def transform_endpoint(request: TransformRequest) -> TransformResponse:
    """Transform the question and query by replacing one entity.
    Args:
        request (TransformRequest): The request body containing question, query, model, lang, and complexity.
    """
    if request.lang not in LANGUAGES:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported language: {request.lang} (supported: {', '.join(LANGUAGES.keys())})"
        )
    
    if request.complexity not in ['easy', 'normal', 'hard', 'random']:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid complexity: {request.complexity} (supported: easy, normal, hard, random)"
        )
    
    if not transform_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="System is busy. Please try again later."
        )
            
    try:
        new_question, new_query, old_pagerank, new_pagerank = create_question_query(
            request.query,
            request.question,
            request.model,
            request.lang,
            request.complexity
        )
        
        return TransformResponse(
            original_question=request.question,
            original_query=request.query,
            transformed_question=new_question,
            transformed_query=new_query,
            old_pagerank=old_pagerank,
            new_pagerank=new_pagerank
        )
    
    except Exception as e:
        logger.error(f"Error in transform endpoint: {e} \n(request: {request})")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Always release the lock so the endpoint becomes available again
        transform_lock.release()


@app.get("/")
def read_root():
    """Redirect to the documentation."""
    return RedirectResponse(url='/docs')


@app.get("/health")
def health_check():
    """Check the health of the API."""
    return {"status": "healthy"}


if DEBUG:
    query = 'SELECT ?answer WHERE { wd:Q14452 wdt:P17 ?answer }'
    create_question_query(
        query,
        'Which country does the famous Easter island belong to?',
        'Gemma3:27b',
        'en', 
        'normal'
    )
else:
    if __name__ == "__main__":
        main()

