import re
# import json
from time import time, sleep

from typing import MutableMapping, Any, Hashable

from collections import defaultdict as ddict
from collections.abc import MutableMapping

from rdflib.plugins.sparql.parser import parseQuery
from rdflib.term import Variable
# from rdflib import Graph, URIRef, Literal, BNode
from rdflib import URIRef
from pyparsing import ParseResults

import requests
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


_last_time = ddict(float)


def wait_time(wait: float, timer_ID: Hashable=None, asynchronous: bool=True) -> bool:
    """
    Wait for a specified time period between events.
    Used for rate limiting and throttling, e.g., for public API calls or resource-intensive operations.
    This function is NOT thread-safe. Do not use in multi-threaded environments.
    
    Args:
        wait: Minimum time interval (in seconds) to wait
        timer_ID: Identifier for the timer. If None, uses default timer
        asynchronous: If False, blocks execution until time condition is met, 
            result immediately otherwise.

    Returns:
        bool: If True, enough time has passed since last call 
            with the same timer_ID (or on the first call), False otherwise.

    Raises:
        TypeError: If wait parameter is not a number
        ValueError: If wait parameter is negative
        TypeError: If timer_ID is not a hashable type
    """
    global _last_time
    
    if not isinstance(wait, (int, float)):
        raise TypeError('wait parameter must be a number')
    if wait < 0:
        raise ValueError('wait parameter cannot be negative')
    if not isinstance(timer_ID, Hashable):
        raise TypeError('timer_ID must be a hashable type')
    
    now = time()

    # Check if this timer has been used before
    if _last_time[timer_ID]:
        # If time condition is not met
        if now - _last_time[timer_ID] < wait:
            if asynchronous:
                return False
            else:
                # Block until enough time has passed
                sleep(max(0, wait - (now - _last_time[timer_ID]))) 
        # If we reach here, enough time has passed
        _last_time[timer_ID] = now  # Update the last time
        return True
    else:
        # First time this timer is used, record the current time
        _last_time[timer_ID] = now
        return True


def timer_reset(timer_ID: Hashable=None) -> bool:
    """
    Reset a specific timer by removing its recorded time.
    This function is NOT thread-safe. Do not use in multi-threaded environments.
    
    Args:
        timer_ID: Identifier of the timer to reset
    Returns:
        bool: True, if the timer was successfully reset
    Raises:
        TypeError: If timer_ID is not a hashable type
    """
    global _last_time
    
    if not isinstance(timer_ID, Hashable):
        raise TypeError('timer_ID must be a hashable type')
    
    # Reset the timer by deleting it (which will be treated as "never used")
    return bool(_last_time.pop(timer_ID, True))


class BiDict(MutableMapping):
    """
    A bidirectional dictionary that stores key: value and value: key pairs.
    Cannot add duplicate values or key:value where key == value.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the BiDict."""
        self.store = dict()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        """Get the value for the given key."""
        return self.store[key]

    def __setitem__(self, key, value):
        """Set the value for the given key."""
        if key == value:
            raise ValueError('Key and value must be different')
        if value in self.store:
            raise ValueError('Value already exists as a key')
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


def execute(query: str, endpoint_url: str, agent: str, delay=2.0, timeout=30.0) -> dict | None:
    """
    Send query direct to endpoint.

    Args:
        query: SPARQL query
        endpoint_url: endpoint of SPARQL query service
        agent: User-Agent string for the request
        delay: guaranteed delay between requests in seconds
    Returns:
        json response from the SPARQL endpoint, or None if an error occurred
    Raises:
        KeyboardInterrupt: If the operation is interrupted by the user
    """
    if delay:
        wait_time(delay, 'wikidata', asynchronous=False)

    try:
        r = requests.get(
            endpoint_url, 
            headers={ 'User-Agent': agent }, 
            params = { 'format': 'json', 'query': query },
            timeout=timeout
        )

        if r.status_code == 200:
            r = r.json()
            return r

    except requests.exceptions.Timeout:
        logger.error(f'Timeout error while executing query: {query}')
    except requests.exceptions.ConnectionError:
        logger.error(f'Connection error while executing query: {query}')
    except requests.exceptions.RequestException as e:
        logger.error(f'Request exception in function "execute": {e}')
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.error(f'Exception in function "execute": {e}')

    return None


def uri2short(resource: str, prefixes: dict=WIKIDATA_PREFIX) -> str:
    """
    Convert entity or predicate URI into short form, e.g. 'wd:Q5'

    Args:
        resource: Full or short URI of the resource. Does nothing if resource is not in full URI form.
        prefixes: dictionary of prefixes to use for conversion. Only prefixes present in this dictionary will be used.
    Returns:
        Short form of the URI, or the original resource if it was not in full URI form.
    """
    if not isinstance(resource, str):
        raise TypeError('Resource must be a string')
    
    if '/' in resource:
        resource = resource.strip()
        # delete angle brackets if they are present
        resource = resource[int(resource.startswith('<')):-int(resource.endswith('>')) or None]
        entity = resource.split('/')[-1].split('#')[-1]
        path = resource[:-(len(entity))]
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


def replace_standard_prefixes(query: str, prefixes: dict=None) -> str:
    """
    Replace all full URIs in the query with their corresponding short forms using the provided prefixes.
    
    Args:
        query: SPARQL query
        prefixes: Dictionary of prefixes to use for conversion

    Returns:
        Query with full URIs replaced by their short forms
    """
    if prefixes is None:
        return query.strip()
    
    def replace_uri(match):
        return uri2short(match.group(0), prefixes)
    
    # Find all URIs in the query (both with and without angle brackets)
    uri_pattern = r'<?(http://[^\s>]+)>?'
    result = re.sub(uri_pattern, replace_uri, query)
    
    return result.strip()


def remove_standard_prefixes_definition(query: str, prefixes: dict=None) -> str:
    """
    Remove PREFIX definitions for standard prefixes from the query's preamble.
    
    Args:
        query: SPARQL query
        prefixes: Dictionary of prefixes to remove from the query
    Returns:
        Query with standard prefix definitions removed from preample
    :rtype: str
    """
    if prefixes is None:
        return query.strip()
    
    def replace_prefix(match):
        # Extract the URL from the PREFIX definition
        url = match.group(0).split(':', 1)[-1].strip(' <>')
        if url in prefixes:
            return ''
        else:
            return match.group(0)
    
    prefix_pattern = r'PREFIX\s+\w+:\s+<[^>]+>'
    return re.sub(prefix_pattern, replace_prefix, query).strip()


def normal_sparql(query: str, prefixes: dict=None) -> str:
    """
    Normalize a SPARQL query by removing standard prefix definitions and replacing full URIs with their short forms.
    
    Args:
        query: SPARQL query.
        prefixes: Dictionary of prefixes to use for conversion.
    Returns:
        Normalized SPARQL query.
    """
    query = remove_standard_prefixes_definition(query, prefixes)
    query = replace_standard_prefixes(query, prefixes)

    REPLACES = (
        (r'\t', r' '),
        (r'\n', r' '),
        (r'  +', r' '),
    )
    for i in REPLACES:
        query = re.sub(*i, query)

    return query


def resolve_item(query_entry, mod=None)->str:
    """
    Resolve the parsed query entry to a string representation.

    Args:
        query_entry: The parsed query entry to resolve.
        mod: Optional modifier to append to the resolved item.
    Returns: 
        String representation of the resolved query entry.
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
    

def parse_query(query: str)-> list[list[str]]:
    """
    Parse a SPARQL query and extract triples from it.

    Args:
        query: The SPARQL query to parse.
    Returns:
        List of triples, where each triple is a list of three strings.
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


def get_levenshtein_distance(a: str, b: str) -> int:
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


def extract_entities_recursive(parsed: Any) -> list[str]:
    """
    Recursively extract entity URIs from parsed SPARQL query.
    
    Args:
        parsed: Parsed SPARQL query object.
    Returns: 
        List of extracted entity URIs as strings.
    """
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


def extract_entities(query: str) -> list[str]:
    """
    Extract all entities from the SPARQL query.

    Args:
        query: SPARQL query string
    Returns: 
        List of extracted entity URIs as strings
    """
    return extract_entities_recursive(parseQuery(query))


def sparql_results_to_list_of_dicts(result: dict) -> list[dict]:
    """
    Convert SPARQL JSON results to a list of dictionaries.

    Args:
        result: SPARQL JSON results
    Returns: 
        List of dictionaries representing the results
    """
    if not result:
        return []
    
    while 'results' in result:
        result = result['results']

    if 'boolean' in result:
        return [{ 'boolean': result['boolean'] }]
    
    result = result.get('bindings', [])

    return [{k: v['value'] if v['type'] == 'literal' else uri2short(v['value'], WIKIDATA_PREFIX) for k, v in i.items()} for i in result]


# def query_wikidata_label(uri: str, endpoint_url, agent: str='', lang: str='en') -> str:
#     """
#     Query the Wikidata label for a given URI.
    
#     :param uri: Wikidata resource URI
#     :type uri: str
#     :param endpoint_url: SPARQL endpoint URL
#     :type endpoint_url: str
#     :param agent: User-Agent string for the request
#     :type agent: str
#     :param lang: Language code for the label (default is 'en')
#     :type lang: str
#     :return: Label for the URI in the specified language, or None if not found
#     :rtype: str
#     """
#     if uri in FIXED_LABELS:
#         return FIXED_LABELS[uri]

#     query = (
#         'SELECT ?label WHERE {',
#             f"OPTIONAL {{ {uri} rdfs:label ?lang_label. FILTER(LANG(?lang_label) = '{lang}') }}",
#             f"OPTIONAL {{ {uri} rdfs:label ?default_label. FILTER(LANG(?default_label) = 'mul') }}",
#             f"OPTIONAL {{ {uri} owl:sameAs+ ?redirect . ?redirect rdfs:label ?redirect_label . FILTER(LANG(?redirect_label) = '{lang}') }}",
#             "BIND(COALESCE(?lang_label, ?default_label, ?redirect_label) AS ?label) . ",
#         '}',
#     )

#     try:
#         data = execute('\n'.join(query), endpoint_url, agent)
#         data = data['results']['bindings'][0]
#         data = data['label']['value']
#     except KeyboardInterrupt:
#         raise
#     except Exception as e:
#         logger.error(f'Exception in function "query_wikidata_label". URI: {uri}, lang: {lang}, error: {e}')
#         return None

#     return data


# def get_wikidata_label(uri: str, endpoint_url: str, agent: str='', lang: str='en', prefixes=WIKIDATA_PREFIX) -> str:
#     """ Get label for a given Wikidata URI.
#     uri: Wikidata resource URI
#     endpoint_url: SPARQL endpoint URL
#     :type endpoint_url: str
#     agent: User-Agent string for the request
#     :type agent: str
#     lang: language code for the label (default is 'en')
#     :type lang: str
#     return: label for the URI in the specified language, or None if not found
#     """
#     try:
#         uri = uri2short(uri, prefixes)
#         prefix = 'wd' # it works despite property or entity
#         index = uri.split(':')[-1]
#         return query_wikidata_label(f'{prefix}:{index}', endpoint_url, agent, lang)
#     except Exception as e:
#         logger.error(f'Exception in function "get_wikidata_label". URI: {uri}, lang: {lang}, error: {e}')
#         return None


def extract_q_number(entity: str) -> int:
    """
    Extract the number from the entity URI (e.g., 'wd:Q123' -> 123).
    
    :param entity: Entity URI string
    :type entity: str
    :return: Extracted number from the entity URI
    :rtype: int
    """
    if entity.startswith('wd:Q'):
        try:
            return int(entity.split(':Q')[-1])
        except ValueError:
            logger.error(f'Exception in function "extract_q_number": ValueError: {entity}')
            return float('inf')
    return float('inf')
