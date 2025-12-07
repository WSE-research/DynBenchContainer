import re
import json
from time import time, sleep

# from glob import glob
# from hashlib import sha1

from collections import defaultdict as ddict
from collections.abc import MutableMapping

from rdflib.plugins.sparql.parser import parseQuery
from rdflib.term import Variable
from rdflib import Graph, URIRef, Literal, BNode
from pyparsing import ParseResults

import requests
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def add_suffix(s: str, suffix: str):
    """Adds suffix to the string s if s is not ending with it, else returns s"""
    return s if len(suffix) == 0 or s.endswith(suffix) else s + suffix


def json_load(name: str, encoding: str='utf-8', suffix='.json'):
    """Load a JSON file from the filesystem."""
    with open(add_suffix(name, suffix), 'r', encoding=encoding) as f:
        return json.load(f)


def json_save(name: str, item, encoding: str='utf-8', indent: int=2, suffix='.json'):
    """Save an item to a JSON file on the filesystem."""
    with open(add_suffix(name, suffix), 'w', encoding=encoding) as f:
        json.dump(item, f, ensure_ascii=False, indent=indent)


def wait_time(wait: float, timer_id: str=None, asynchronous=True) -> bool:
    """helps to wait particular time period between events"""
    global last_time
    
    now = time()

    if last_time[timer_id]:
        while now - last_time[timer_id] < wait:
            if asynchronous:
                return False
            else:
                sleep(0.1)
    else:
        last_time[timer_id] = now

    return True


last_time = ddict(float)


def timer_reset(timer_id: str):
    last_time[timer_id] = None


class BiDict(MutableMapping):
    def __init__(self, *args, **kwargs):
        """Initialize the BiDict."""
        self.store = dict()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        """Get the value for the given key."""
        return self.store[key]

    def __setitem__(self, key, value):
        """Set the value for the given key."""
        self.store[key] = value
        self.store[value] = key

    def __delitem__(self, key):
        """Delete the value for the given key."""
        del self.store[self.store[key]]
        del self.store[key]

    def __iter__(self):
        """Iterate over the keys in the BiDict."""
        return iter(self.store)
    
    def __len__(self):
        """Get the length of the BiDict."""
        return len(self.store) // 2


WIKIDATA_PREFIX = BiDict({
    'bd': 'http://www.bigdata.com/rdf#',
    'cc': 'http://creativecommons.org/ns#',
    'dct': 'http://purl.org/dc/terms/',
    'geo': 'http://www.opengis.net/ont/geosparql#',
    'ontolex': 'http://www.w3.org/ns/lemon/ontolex#',
    'owl': 'http://www.w3.org/2002/07/owl#',
    'p': 'http://www.wikidata.org/prop/',
    'pq': 'http://www.wikidata.org/prop/qualifier/',
    'pqn': 'http://www.wikidata.org/prop/qualifier/value-normalized/',
    'pqv': 'http://www.wikidata.org/prop/qualifier/value/',
    'pr': 'http://www.wikidata.org/prop/reference/',
    'prn': 'http://www.wikidata.org/prop/reference/value-normalized/',
    'prov': 'http://www.w3.org/ns/prov#',
    'prv': 'http://www.wikidata.org/prop/reference/value/',
    'ps': 'http://www.wikidata.org/prop/statement/',
    'psn': 'http://www.wikidata.org/prop/statement/value-normalized/',
    'psv': 'http://www.wikidata.org/prop/statement/value/',
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
    'schema': 'http://schema.org/',
    'skos': 'http://www.w3.org/2004/02/skos/core#',
    'wd': 'http://www.wikidata.org/entity/',
    'wdata': 'http://www.wikidata.org/wiki/Special:EntityData/',
    'wdno': 'http://www.wikidata.org/prop/novalue/',
    'wdref': 'http://www.wikidata.org/reference/',
    'wds': 'http://www.wikidata.org/entity/statement/',
    'wdt': 'http://www.wikidata.org/prop/direct/',
    'wdtn': 'http://www.wikidata.org/prop/direct-normalized/',
    'wdv': 'http://www.wikidata.org/value/',
    'wikibase': 'http://wikiba.se/ontology#',
    'xsd': 'http://www.w3.org/2001/XMLSchema#',
})


FIXED_LABELS = {
    'rdfs:label': 'label',
    'http://www.w3.org/2000/01/rdf-schema#label': 'label',
    'skos:altLabel': 'alternative label',
    'xsd:integer': 'integer',
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#type': 'type',
    'http://www.w3.org/2001/XMLSchema#integer': 'integer',
    'http://www.w3.org/2001/XMLSchema#gYear': 'year',
    'http://www.w3.org/2001/XMLSchema#double': 'double',
}


def execute(query: str, endpoint_url: str, agent: str, delay=2.0, cache=None):
    """
    Check cache first, then send query direct to wikidata.
    query: SPARQL query
    endpoint_url: endpoint of wikidata query service
    agent: User-Agent string for the request
    delay: guaranteed delay between requests in seconds
    cache: optional MongoDB collection for caching results
    return: json response
    """
    if cache is not None:
        r = cache.find_one({ 'query': query, })
        if r:
            return r['result']

    if delay:
        wait_time(delay, 'wikidata', asynchronous=False)

    headers = {
        'User-Agent': agent
    }
            
    try:
        r = requests.get(endpoint_url, headers=headers, params = { 'format': 'json', 'query': query })

        if r.status_code == 200:
            r = r.json()
            if cache is not None:
                cache.insert_one({ 'query': query, 'result': r })
            return r

    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.error(f'Exception in function "execute": {e}')

    return None


def uri2short(resource: str, prefixes=WIKIDATA_PREFIX):
    """convert entity or predicate URI into short form, e.g. 'wd:Q5'"""
    if '/' in resource:
        resource = resource.strip()
        # delete angle brackets if they are present
        resource = resource[int(resource.startswith('<')):-int(resource.endswith('>')) or None]
        entity = resource.split('/')[-1]
        path = '/'.join(resource.strip().split('/')[:-1])
        # ensure that path ends with '/' or '#'
        path = path if (path[-1] in {'/', '#'}) else path + '/'

        if prefixes and path in prefixes:
            uri =  f'{prefixes[path]}:{entity}'
        else:
            uri = f'<{path}{entity}>'

        return uri
    else:
        # if resource is not a full URI, return it as is
        return resource
    

def find_prefixes(query):
    """Find the prefixes in the SPARQL query."""
    prefix_pattern = r'PREFIX\s+\w+:\s+<[^>]+>'
    return re.findall(prefix_pattern, query)


def find_uris(sparql_query):
    """Find the URIs in the SPARQL query."""
    uri_pattern = r'\<*http://[^\s>]+\>*'
    uris = re.findall(uri_pattern, sparql_query)
    return uris


def replace_standard_prefixes(sparql, prefixes=None):
    """Replace the standard prefixes in the SPARQL query."""
    body = sparql.split('WHERE')[-1]
    uris = find_uris(body)

    for uri in uris:
        short = uri2short(uri, prefixes)
        sparql = sparql.replace(uri, short)

    return sparql.strip()


def remove_standard_prefixes(sparql):
    """Remove the standard prefixes from the SPARQL query."""
    head = sparql.split('WHERE')[0]
    prefixes = find_prefixes(head)

    for prefix in prefixes:
        sparql = sparql.replace(prefix, '')

    return sparql.strip()


def normal_sparql(query: str, prefixes=None) -> str:
    """Normalize the SPARQL query."""
    query = remove_standard_prefixes(query)
    if prefixes:
        query = replace_standard_prefixes(query, prefixes)

    REPLACES = (
        (r'\t', r' '),
        (r'\n', r' '),
        (r'  +', r' '),
    )

    for i in REPLACES:
        query = re.sub(*i, query)

    return query


def resolve_item(query_entry, mod=None):
    """
    Resolve the parsed query entry to a string representation.
    Args:
        query_entry: The parsed query entry to resolve.
        mod: Optional modifier to append to the resolved item.
    Returns:
        str: The resolved item as a string.
    """
    if isinstance(query_entry, Variable):
        return f'?{query_entry}'        
    
    if isinstance(query_entry, dict):
        if 'part' in query_entry:
            if 'mod' in query_entry:
                mod = query_entry['mod']
            return resolve_item(query_entry['part'], mod)
        elif 'prefix' in query_entry and 'localname' in query_entry:
            if mod:
                return query_entry['prefix'] + ':' + query_entry['localname'] + mod
            else:
                return query_entry['prefix'] + ':' + query_entry['localname']
    elif isinstance(query_entry, list):
        return resolve_item(query_entry[0])
    

def parse_query(query):
    """
    Parse a SPARQL query and extract triples from it.
    Args:
        query (str): The SPARQL query to parse.
    Returns:
        List[List[str]]: A list of triples, where each triple is a list of three strings.
    """
    parsed = parseQuery(query)
    select_query = parsed[1]
    where = select_query['where']           
    group_parts = where['part']             

    triples = []
    for part in group_parts:
        if part.name == 'TriplesBlock':
            for triple in part['triples']:
                triples.append([
                        resolve_item(triple[0]), 
                        resolve_item(triple[1]), 
                        resolve_item(triple[2])
                    ])

    return triples


def resolve_item(query_entry, mod=None):
    """
    Resolve the parsed query entry to a string representation.
    Args:
        query_entry: The parsed query entry to resolve.
        mod: Optional modifier to append to the resolved item.
    Returns:
        str: The resolved item as a string.
    """
    if isinstance(query_entry, Variable):
        return f'?{query_entry}'        
    
    if isinstance(query_entry, dict):
        if 'part' in query_entry:
            if 'mod' in query_entry:
                mod = query_entry['mod']
            return resolve_item(query_entry['part'], mod)
        elif 'prefix' in query_entry and 'localname' in query_entry:
            if mod:
                return query_entry['prefix'] + ':' + query_entry['localname'] + mod
            else:
                return query_entry['prefix'] + ':' + query_entry['localname']
    elif isinstance(query_entry, list):
        return resolve_item(query_entry[0])
    

def parse_query(query):
    """
    Parse a SPARQL query and extract triples from it.
    Args:
        query (str): The SPARQL query to parse.
    Returns:
        List[List[str]]: A list of triples, where each triple is a list of three strings.
    """
    parsed = parseQuery(query)
    select_query = parsed[1]
    where = select_query['where']           
    group_parts = where['part']             

    triples = []
    for part in group_parts:
        if part.name == 'TriplesBlock':
            for triple in part['triples']:
                triples.append([
                        resolve_item(triple[0]), 
                        resolve_item(triple[1]), 
                        resolve_item(triple[2])
                    ])

    return triples


def get_levenshtein_distance(a, b) -> int:
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


def extract_entities_recursive(parsed):
    """Extract the entities from the parsed SPARQL query."""
    results = []

    if isinstance(parsed, URIRef):
        results += [str(parsed)]

    if isinstance(parsed, (list, ParseResults)):
        for i in list(parsed):
            results += extract_entities_recursive(i)

    if isinstance(parsed, dict):
        if 'prefix' in parsed and 'localname' in parsed:
            results += [f'{parsed["prefix"]}:{parsed["localname"]}']
        else:
            for i in parsed.values():
                results += extract_entities_recursive(i)

    return results


def extract_entities(sparql):
    """Extract the entities from the SPARQL query."""
    return extract_entities_recursive(parseQuery(sparql))


def sparql_results_to_list_of_dicts(result):
    """Convert the SPARQL results to a list of dictionaries."""
    if not result:
        return[]
    
    while 'results' in result:
        result = result['results']

    if 'boolean' in result:
        return [{'boolean': result['boolean']}]
    
    result = result.get('bindings', [])

    result = [{k: v['value'] if v['type'] == 'literal' else uri2short(v['value'], WIKIDATA_PREFIX) for k, v in i.items()} for i in result]
    
    return result


def query_wikidata_label(uri: str, endpoint_url, agent, lang: str='en', cache=None) -> str:
    """Query the Wikidata label for a given URI."""
    if uri in FIXED_LABELS:
        return FIXED_LABELS[uri]

    query = (
        'SELECT ?label WHERE {',
            f"OPTIONAL {{ {uri} rdfs:label ?lang_label. FILTER ( lang(?lang_label) = '{lang}' ) }}",
            f"OPTIONAL {{ {uri} rdfs:label ?default_label. FILTER ( lang(?default_label) = 'mul' ) }}",
            f"OPTIONAL {{ {uri} owl:sameAs+ ?redirect . ?redirect rdfs:label ?label . FILTER ( lang(?lang_label) = '{lang}' ) }}",
            "BIND(COALESCE(?lang_label, ?default_label) AS ?label) . ",
        '}',
    )

    try:
        data = execute('\n'.join(query), endpoint_url, agent, cache=cache)
        data = data['results']['bindings'][0]
        data = data['label']['value']
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.error(f'Exception in function "query_wikidata_label". URI: {uri}, lang: {lang}, error: {e}')
        return None

    return data


def get_wikidata_label(uri: str, endpoint_url, agent, lang: str='en', cache=None) -> str:
    """ Get label for a given Wikidata URI.
    uri: Wikidata URI
    lang: language code for the label (default is 'en')
    cache: optional cache object to store and retrieve labels
        cache must have methods `find_one` and `insert_one`
    return: label for the URI in the specified language, or None if not found
    """
    prefix = 'wd' # it works despite property or entity
    index = uri.split(':')[-1]

    return query_wikidata_label(f'{prefix}:{index}', endpoint_url, agent, lang, cache)


def check_productivity_single(query, replace, endpoint_url, agent, cache=None):
    """Check if the query is productive after replacing the old entity with the new one."""
    sparql = query.replace(replace['old'], replace['new']).strip()
    try:
        result = sparql_results_to_list_of_dicts(execute(sparql, endpoint_url, agent, cache=cache))
        return bool(result)
    except Exception as e:
        logger.error(f'Exception in function "check_productivity_single": {e}')
        return False
    

def extract_q_number(entity):
    """Extract the Q number from the entity."""
    if entity.startswith('wd:Q'):
        try:
            return int(entity.split(':Q')[-1])
        except ValueError:
            logger.error(f'Exception in function "extract_q_number": ValueError: {entity}')
            return float('inf')
    return float('inf')
