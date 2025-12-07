#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import random
import argparse
from collections import defaultdict as dd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import RedirectResponse

from pymongo import MongoClient
from pymongo.database import Database

import logging

from decouple import Config, RepositoryEnv

import nltk

from nltk.tokenize import sent_tokenize

from dynutils import execute, get_wikidata_label, sparql_results_to_list_of_dicts
from dynutils import parse_query, extract_entities
from dynutils import extract_q_number, check_productivity_single
from dynutils import get_levenshtein_distance


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


config = Config(RepositoryEnv('.env'))

MONGO_HOST = config('MONGO_HOST')
MONGO_USER = config('MONGO_USER')
MONGO_PASS = config('MONGO_PASS')

LLM_URL = config('LLM_URL')

WIKIDATA_AGENT = config('WIKIDATA_AGENT')
WIKIDATA_ENDPOINT = config('WIKIDATA_ENDPOINT')

KEY = config('KEY')

mongo = MongoClient(
    MONGO_HOST,
    username=MONGO_USER,
    password=MONGO_PASS,
)

db = mongo['wikidata']
cache = db['cache']

page_rank = {}

with open('pagerank/2025-11-05.allwiki.links.rank', 'r') as f:
    for line in f:
        entity, rank = line.split('\t')
        page_rank[entity.strip()] = int(float(rank.strip())*100)

LANGUAGES = {
    'en': 'English',
    'de': 'German',
    'fr': 'French',
    'ru': 'Russian',
    'uk': 'Ukrainian',
}

PREDICATES = ('wdt:P31', 'wdt:P279', 'wdt:P106')


def get_resources_types(item, cache=None, predicates=[]):
    """Get the types of the resources in the item."""
    results = {}

    if not predicates:
        return results

    for entity in item['resources']:
        if not entity.startswith('wd:'):
            continue

        query = 'SELECT DISTINCT ?p ?o WHERE { VALUES ?p { ' + ' '.join(predicates) + ' } ' + entity + ' ?p ?o }'

        try:
            r = execute(query, WIKIDATA_ENDPOINT, WIKIDATA_AGENT, cache=cache)
            r = sparql_results_to_list_of_dicts(r)
            r = {p: [v['o'] for v in r if v['p'] == p] for p in predicates}
            for i in predicates:
                r[i].sort(key=extract_q_number)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.warning(f'Skipped exception in function get_resources_types: {e}')
            continue

        results[entity] = r
        
    return results


def get_conditions_by_predicates(item, predicates):
    conditions = dd(dict)
    
    for entity in item['resources']:
        if not entity.startswith('wd:'):
            continue

        for predicate in predicates:
            properties = item['types'][entity][predicate]
            conditions[entity][predicate] = [f'?subst {predicate} {i}' for i in properties]

    return conditions


def get_query_conditions(item):
    """
    get conditions from the query, so new entities will be connected to the same properties
    """
    query_conditions = {}

    for entity in item['resources']:
        if not entity.startswith('wd:'):
            continue

        query_conditions[entity] = []

        v = 0
        for triple in item['triples']:
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


def find_substitutes(query, info, cache=None):
    """
    Find possible substitutes for the each entity in the item.
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
            result = execute(extract_query, WIKIDATA_ENDPOINT, WIKIDATA_AGENT, cache=cache)
            result = sparql_results_to_list_of_dicts(result)
            for r in result:
                r['old'] = entity
            substitutes.append({ 'query': query, 'entity': entity, 'extract': extract_query, 'results': result })
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f'Exception in function find_substitutes: {e}')
            continue

    return substitutes


def call_LLM(url, key, model, prompt, temp=0.0, max_tokens=1000):
    """Call the Ollama API to generate a response based on the provided model and prompt."""   
    headers = {
        'content-type': 'application/json',
        'Authorization': f'Bearer {key}',
    }
    data = {
        'model': model,
        'prompt': prompt,
        'stream': False,
        'options': {
            'temperature': temp,
            'num_predict': max_tokens,
        }
    }
    response = requests.post(url, json=data, headers=headers)

    return response.json() if response.status_code == 200 else None


def replace_entity(model, question, query, entity, new_entity, lang='en'):
    """
    Replace the entity in the question and query with a new entity.
    """
    old_label= get_wikidata_label(uri=entity, endpoint_url=WIKIDATA_ENDPOINT, agent=WIKIDATA_AGENT, lang=lang, cache=cache)
    new_label= get_wikidata_label(uri=new_entity, endpoint_url=WIKIDATA_ENDPOINT, agent=WIKIDATA_AGENT, lang=lang, cache=cache)

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
    
    new_question = call_LLM(LLM_URL, KEY, model, prompt, 0.0)

    if not new_question:
        return None, None
    
    return new_query, new_question['response']
    

def build_pagerank_list(substitutes):
    """
    creates list of tuples for later sorting:
        original pagerank
        new pagerank
        replace dict
    """
    result = []

    for r in substitutes:
        old = r['old'].split(':')[-1]
        new = r['new'].split(':')[-1]
        if old in page_rank and new in page_rank:
            result.append((page_rank[old], page_rank[new], r))

    return result


def count_sentences(s: str) -> int:
    """Count the number of sentences in the string."""
    n = sent_tokenize(s)
    return len(n)


def get_info(query: str, cache=None):
    """Get the information about the query."""
    info = {}
    info['triples'] = [i for i in parse_query(query) if all(i)]
    info['resources'] = extract_entities(query)
    info['types'] = get_resources_types(info, cache, PREDICATES)
    info['conditions'] = get_conditions_by_predicates(info, PREDICATES)
    info['query conditions'] = get_query_conditions(info)
    info['substitutes'] = find_substitutes(query, info, cache=cache)

    all_replaces = []
    for sub in info['substitutes']:
        if 'results' in sub:
            all_replaces += [i | { 'old': sub['entity'] } for i in sub['results']]

    unic_replaces = {tuple((k, v) for k, v in i.items() if k in {'old', 'subst'}) for i in all_replaces}
    unic_replaces = [dict(i) for i in unic_replaces]
    for u in unic_replaces:
        u['new'] = u.pop('subst')

    for r in unic_replaces:
        for a in all_replaces:
            if a['old'] == r['old'] and a['subst'] == r['new']:
                r[a['lang']] = { 'label': a['label'] }

    info['substitutes'] = unic_replaces

    # Add page rank for all entities
    for sub in info['substitutes']:
        o = sub['old'].split(':')[-1]
        n = sub['new'].split(':')[-1]
        sub['old pagerank'] = page_rank.get(o, None)
        sub['new pagerank'] = page_rank.get(n, None)

    return info


def sort_replaces_by_complexity(replaces, complexity):
    """
    Sort or shuffle the replaces list based on the complexity level.
    
    Args:
        replaces: List of tuples (original_pagerank, new_pagerank, replace_dict)
        complexity: One of 'easy', 'normal', 'hard', or 'random'
    
    Returns:
        Sorted or shuffled list of replaces
    
    Raises:
        ValueError: If complexity is not one of the valid options
    """
    if complexity == 'easy':
        return sorted(replaces, key=lambda x: x[1], reverse=True)  # easy
    elif complexity == 'normal':
        return sorted(replaces, key=lambda x: abs(x[0]-x[1]))  # normal
    elif complexity == 'hard':
        return sorted(replaces, key=lambda x: x[1])  # hard
    elif complexity == 'random':
        shuffled = replaces.copy()
        random.shuffle(shuffled)
        return shuffled
    else:
        raise ValueError(f'Complexity can only be easy/normal/hard/random. Got: {complexity}')


def create_question_query(query, question, model, lang, complexity, cache=None):
    """Create a new question and query by replacing one entity."""
    info = get_info(query, cache)

    replaces = build_pagerank_list(info['substitutes'])
    replaces = sort_replaces_by_complexity(replaces, complexity)

    for replace in replaces:
        replace = replace[2]

        old_label = get_wikidata_label(replace['old'], WIKIDATA_ENDPOINT, WIKIDATA_AGENT, lang=lang, cache=cache)
        if not old_label:
            return None, None

        if not check_productivity_single(query, replace, WIKIDATA_ENDPOINT, WIKIDATA_AGENT, cache=cache):
            continue

        new_question = None
        new_query = None

        try:
            new_query, new_question = replace_entity(model, question, query, replace['old'], replace['new'], lang)
            if not new_question or not new_query:
                continue

            new_question = new_question.strip(' ,\n\t')

            old_len = count_sentences(question)
            new_len = count_sentences(new_question)

            if new_len != old_len:
                continue

            _, restored_question = replace_entity(model, new_question, new_query, replace['new'], replace['old'], lang)
            restored_question = restored_question.strip(' ,\n\t')
            dist =  get_levenshtein_distance(question, restored_question)
            if dist > 4:
                continue

            return new_question, new_query
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f'Error in create_new_question_query. Question: {question}. Query: {query}. Languaage: {lang}. Error: {e}.')
            continue
        
    return None, None
        

# DEBUG
# query = 'SELECT ?answer WHERE { wd:Q14452 wdt:P17 ?answer }'
# create_question_query(
#     query,
#     'Which country does the famous Easter island belong to?',
#     'Gemma3:27b',
#     'en', 
#     'normal',
#     cache,
# )

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

    logger.info('Loading PageRank files...')
    try:
        with open('pagerank/2025-11-05.allwiki.links.rank', 'r') as f:
            for line in f:
                entity, rank = line.split('\t')
                page_rank[entity.strip()] = int(float(rank.strip())*100)
            logger.info('PageRank file loaded successfully')
    except Exception as e:
        logger.error(f'Error loading PageRank file in function main of dynbench.py: {e}. Exiting...')
        exit(1)

    new_question, new_query = create_question_query(query, question, model, lang, complexity, cache)

    print('   Original Question:', question)
    print('      Original Query:', query)
    print('Transformed Question:', new_question)
    print('   Transformed Query:', new_query)


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


@app.post("/transform", response_model=TransformResponse)
def transform_endpoint(request: TransformRequest):
    if request.lang not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {request.lang} (supported: {', '.join(LANGUAGES.keys())})")
    
    if request.complexity not in ['easy', 'normal', 'hard', 'random']:
        raise HTTPException(status_code=400, detail=f"Invalid complexity: {request.complexity} (supported: easy, normal, hard, random)")
    
    try:
        new_question, new_query = create_question_query(
            request.query,
            request.question,
            request.model,
            request.lang,
            request.complexity,
            cache
        )
        
        return TransformResponse(
            original_question=request.question,
            original_query=request.query,
            transformed_question=new_question,
            transformed_query=new_query
        )
    
    except Exception as e:
        logger.error(f"Error in transform endpoint: {e} (request: {request})")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/")
def read_root():
    """Redirect to the documentation."""
    return RedirectResponse(url='/docs')


@app.get("/health")
def health_check():
    """Check the health of the API."""
    return {"status": "healthy"}


if __name__ == "__main__":
    main()

    ## dynbench.py
