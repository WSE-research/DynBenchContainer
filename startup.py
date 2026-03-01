import logging

logger = logging.getLogger(__name__)

import requests

from decouple import config

from pymongo import MongoClient

from utils.timer import wait_time

from utils.mongocache import MongoCache

from utils.sparql import execute as raw_execute

from utils.wikidata import get_wikidata_label

from utils.llm import call_LLM as raw_call_LLM


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


@cache.cache(using={'query'})
def execute(query: str, delay=2.0, timeout=30.0,) -> dict | None:
    return raw_execute(query, WIKIDATA_ENDPOINT, WIKIDATA_AGENT, delay=delay, timeout=timeout)


@cache.cache(using={'model', 'prompt', 'temp', 'max_tokens'})
def call_LLM(url: str, key: str, model: str, prompt, temp: float=0.0, max_tokens: int=1000, timeout=30.0) -> dict | None:
    return raw_call_LLM(url, key, model, prompt, temp, max_tokens, timeout)
        

def get_label(entity: str, lang: str='en') -> str:
    return get_wikidata_label(entity, execute, lang=lang)


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
    logger.error(f'Error loading PageRank file in settings.py: {e}. Exiting...')
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

