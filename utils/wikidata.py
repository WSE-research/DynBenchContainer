from typing import Callable
import logging

from .sparql import uri2short
from .text import extract_number

from .sparql import sparql_results_to_list_of_dicts


logger = logging.getLogger(__name__)


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


# standard wikidata prefixes
WIKIDATA_PREFIX = {
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
}

# reverse order for standard prefixes
for k, v in list(WIKIDATA_PREFIX.items()):
    WIKIDATA_PREFIX[v] = k

def query_wikidata_label(
        uri: str, 
        execute: Callable,
        lang: str='en', 
        fixed_labels: dict=None,
    ) -> str:
    """
    Query the Wikidata label for a given URI.
    
    Args:
        uri: Wikidata resource URI
        endpoint_url: SPARQL endpoint URL
        agent: User-Agent string for the request
        lang: Language code for the label (default is 'en')
    Returns:
        Label for the URI in the specified language, or None if not found
    """
    if fixed_labels and uri in fixed_labels:
        return fixed_labels[uri]

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


def get_wikidata_label(
        uri: str, 
        execute: Callable, 
        lang: str='en', 
        prefixes: dict=WIKIDATA_PREFIX
    ) -> str:
    """ Get label for a given Wikidata URI.

    Args:
        uri: Wikidata resource URI
        endpoint_url: SPARQL endpoint URL
        agent: User-Agent string for the request
        lang: language code for the label (default is 'en')
    Returns:
        label for the URI in the specified language, or None if not found
    """
    try:
        uri = uri2short(uri, prefixes)
        prefix = 'wd' # it works despite property or entity
        index = uri.split(':')[-1]
        return query_wikidata_label(f'{prefix}:{index}', execute, lang, prefixes)
    except Exception as e:
        logger.error(f'Exception in function "get_wikidata_label". URI: {uri}, lang: {lang}, error: {e}')
        return None


def get_resources_types(info: dict, execute: Callable, predicates: list=[]) -> dict:
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
            r = sparql_results_to_list_of_dicts(r, WIKIDATA_PREFIX)
            r = {p: [v['o'] for v in r if v['p'] == p] for p in predicates}
            for i in predicates:
                r[i].sort(key=extract_number)

            logger.debug(f'Collected properties for entity {entity}.')
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f'Exception {e} in function get_resources_types for entity {entity}.')

        results[entity] = r
        
    return results


def find_substitutes(query: str, execute: Callable, info: dict) -> list:
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
            result = sparql_results_to_list_of_dicts(result, WIKIDATA_PREFIX)

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


def check_productivity_single(query: str, execute: Callable, replace: dict, prefixes: dict = WIKIDATA_PREFIX) -> bool:
    """
    Check if query is productive (i.e., returns non-empty results).
    
    Args:
        query: SPARQL query
        execute: Function to execute SPARQL queries
        replace: Dictionary containing 'old' and 'new' keys for string replacement in the query
        prefixes: Dictionary of prefixes for SPARQL queries (default is WIKIDATA_PREFIX)
    Returns:
        True if the query returns non-empty results, False otherwise
    """
    sparql = query.replace(replace.get('old', ''), replace.get('new', '')).strip()
    try:
        result = sparql_results_to_list_of_dicts(execute(sparql), prefixes)
        return bool(result)
    except Exception as e:
        logger.error(f'Exception in function "check_productivity_single": {e}')
        return False