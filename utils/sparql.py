import re
import requests

from rdflib.plugins.sparql.parser import parseQuery
from rdflib.term import Variable
from rdflib import URIRef
from pyparsing import ParseResults

import logging

from .timer import wait_time


logger = logging.getLogger(__name__)


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


def sparql_results_to_list_of_dicts(result: dict, prefixes) -> list[dict]:
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

    return [{k: v['value'] if v['type'] == 'literal' else uri2short(v['value'], prefixes) for k, v in i.items()} for i in result]


def uri2short(resource: str, prefixes) -> str:
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


def extract_entities_recursive(parsed) -> list[str]:
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

