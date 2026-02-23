#!/usr/bin/env python3
# -*- coding: utf-8 -*-

DEBUG = False

import traceback

import requests
import random
import argparse
from collections import defaultdict as ddict

import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import RedirectResponse

from pymongo import MongoClient

import logging
from uvicorn.config import LOGGING_CONFIG

from decouple import config

from utils.timer import wait_time
from utils.mongocache import MongoCache

from utils.sparql import execute as raw_execute
from utils.sparql import normal_sparql, parse_query, extract_entities

from utils.wikidata import get_wikidata_label, WIKIDATA_PREFIX
from utils.wikidata import get_resources_types, find_substitutes
from utils.wikidata import check_productivity_single

from utils.text import count_sentences, calc_levenshtein_dist

from utils.llm import call_LLM as raw_call_LLM


logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('uvicorn.error')
if DEBUG:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)


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
feedback_collection = db.feedback

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
    
    
# Load PageRank file into memory
page_rank = {}

wait_time(0.0, 'pagerank file load') # init timer to skip "Loaded 1 record" message
logger.info('Loading PageRank file...')
try:
    with open('pagerank/allwiki.rank', 'r') as f:
        for x, line in enumerate(f):
            entity, rank = line.split('\t')
            page_rank[entity.strip()] = float(rank.strip())
            if wait_time(1.0, 'pagerank file load'):
                logger.info(f'Loaded {x+1:,} records.'.replace(',', ' '))
    logger.info('PageRank file loaded successfully')
except Exception as e:
    logger.error(f'Error loading PageRank file in dynbench.py: {e}. Exiting...')
    exit(1)

# Top 20 languages by number of speakers in Europe
LANGUAGES = {
    'English':    'en',
    'German':     'de',
    'French':     'fr',
    'Russian':    'ru',
    'Ukrainian':  'uk',
    'Italian':    'it',
    'Spanish':    'es',
    'Polish':     'pl',
    'Romanian':   'ro',
    'Dutch':      'nl',
    'Turkish':    'tr',
    'Bavarian':   'bar',
    'Portuguese': 'pt',
    'Hungarian':  'hu',
    'Greek':      'el',
    'Czech':      'cs',
    'Swedish':    'sv',
    'Catalan':    'ca',
    'Serbian':    'sr',
    'Bulgarian':  'bg',
}

# Add reverse order
for k, v in list(LANGUAGES.items()):
    LANGUAGES[v] = k

PREDICATES = ('wdt:P31', 'wdt:P279', )


@cache.cache(using={'query'})
def execute(query: str, delay=2.0, timeout=30.0,) -> dict | None:
    return raw_execute(query, WIKIDATA_ENDPOINT, WIKIDATA_AGENT, delay=delay, timeout=timeout)


def get_label(entity: str, lang: str='en') -> str:
    return get_wikidata_label(entity, execute, lang=lang)


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


@cache.cache(using={'model', 'prompt', 'temp', 'max_tokens'})
def call_LLM(url: str, key: str, model: str, prompt, temp: float=0.0, max_tokens: int=1000, timeout=30.0) -> dict | None:
    return raw_call_LLM(url, key, model, prompt, temp, max_tokens, timeout)


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

    info['types'] = get_resources_types(info, execute, PREDICATES)
    logger.debug(f'Extracted entity properties.')

    info['conditions'] = get_conditions_by_predicates(info, PREDICATES)
    logger.debug(f'Extracted conditions.')

    info['query conditions'] = get_query_conditions(info)
    logger.debug(f'Extracted query conditions.')

    info['substitutes'] = find_substitutes(query, execute, info)
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
        Tuple of (new_question, new_query) or (None, None) if no valid replacement found.
    Raises:
        Exception: If an error occurs during processing.
    """
    logger.info(f'Transforming question: "{question}"...')

    if checks is None:
        checks = {}
    elif isinstance(checks, (tuple, list)):
        checks = set(checks)
    elif not isinstance(checks, set):
        logger.error('Value error in create_question_query: checks must be of type tuple, list, set or None')
        return None, None, None, None

    query = normal_sparql(query, WIKIDATA_PREFIX)
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

        if not check_productivity_single(query, execute, replace):
            continue

        new_question = None
        new_query = None
        num_tries = 0

        try:
            num_tries += 1
            # !TODO if label of item is different from used in the question then all back-transforms are failing
            #       set a max number of repeats for one entity
            if num_tries > 6:
                return None, None, None, None

            logger.info(f'Replace {old_label} -> {new_label}.')
            new_query, new_question = replace_entity(model, question, query, replace['old'], replace['new'], lang)
            if not new_question or not new_query:
                continue

            new_question = new_question.strip(' ,\n\t')

            old_len = count_sentences(question)
            new_len = count_sentences(new_question)

            logger.info(f'New question: {new_question}')

            if checks and 'sentence' in checks and new_len != old_len:
                logger.info(f'Sentence count check failed (changed from {old_len} to {new_len}). Skipping replacement.')
                continue

            logger.info(f'Back-transform {new_label} -> {old_label}.')
            _, restored_question = replace_entity(model, new_question, new_query, replace['new'], replace['old'], lang)
            restored_question = restored_question.strip(' ,\n\t')
            logger.info(f'Back-transformed question: {restored_question}')

            dist = calc_levenshtein_dist(question, restored_question)
            if checks and 'levenstein' in checks and dist > 4:
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
        choices=[v for v in LANGUAGES.values() if len(v) < 4],
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

    new_question, new_query, old_pagerank, new_pagerank = create_question_query(query, question, model, lang, complexity, None)

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
    checks: list[str] | None


class TransformResponse(BaseModel):
    original_question: str
    original_query: str
    transformed_question: str | None
    transformed_query: str | None
    old_pagerank: float | None
    new_pagerank: float | None


class FeedbackRequest(BaseModel):
    inputs: list[str]
    outputs: list[str]
    rating: int


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
            request.complexity,
            request.checks
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


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    try:
        feedback_doc = {
            "inputs": feedback.inputs,
            "outputs": feedback.outputs,
            "rating": feedback.rating,
        }

        result = await feedback_collection.insert_one(feedback_doc)

        return {
            "message": "Feedback stored successfully",
            "feedback_id": str(result.inserted_id)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        'normal',
        None
    )
else:
    if __name__ == "__main__":
        main()

