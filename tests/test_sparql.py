# import pytest
from unittest.mock import patch, MagicMock

import requests

from utils.sparql import (
    execute,
    sparql_results_to_list_of_dicts,
    uri2short,
    replace_standard_prefixes,
    remove_standard_prefixes_definition,
    normal_sparql,
    resolve_item,
    parse_query,
    extract_entities_recursive,
    extract_entities
)
from rdflib import URIRef
# from rdflib.plugins.sparql.parser import ParseResults


class TestExecute:
    @patch('utils.sparql.requests.get')
    @patch('utils.sparql.wait_time')
    def test_execute_success(self, mock_wait, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'results': {'bindings': []}}
        mock_get.return_value = mock_response
        
        result = execute('SELECT * WHERE {?s ?p ?o}', 'http://example.com/sparql', 'TestAgent', delay=0)
        assert result == {'results': {'bindings': []}}
        mock_get.assert_called_once()
    
    @patch('utils.sparql.requests.get')
    @patch('utils.sparql.wait_time')
    def test_execute_timeout(self, mock_wait, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout()
        
        result = execute('SELECT * WHERE {?s ?p ?o}', 'http://example.com/sparql', 'TestAgent', delay=0)
        assert result is None
    
    @patch('utils.sparql.requests.get')
    @patch('utils.sparql.wait_time')
    def test_execute_connection_error(self, mock_wait, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError()
        
        result = execute('SELECT * WHERE {?s ?p ?o}', 'http://example.com/sparql', 'TestAgent', delay=0)
        assert result is None


class TestSparqlResultsToListOfDicts:
    def test_empty_result(self):
        result = sparql_results_to_list_of_dicts(None, {})
        assert result == []
    
    def test_boolean_result(self):
        result = sparql_results_to_list_of_dicts({'boolean': True}, {})
        assert result == [{'boolean': True}]
    
    def test_bindings_result(self):
        prefixes = {'http://example.com/': 'ex'}
        result = {
            'results': {
                'bindings': [
                    {
                        's': {'type': 'uri', 'value': 'http://example.com/resource1'},
                        'p': {'type': 'literal', 'value': 'test'}
                    }
                ]
            }
        }
        converted = sparql_results_to_list_of_dicts(result, prefixes)
        assert converted == [{'s': 'ex:resource1', 'p': 'test'}]


class TestUri2short:
    def test_full_uri_conversion(self):
        prefixes = {'http://www.wikidata.org/entity/': 'wd'}
        result = uri2short('http://www.wikidata.org/entity/Q5', prefixes)
        assert result == 'wd:Q5'
    
    def test_full_uri_with_angle_brackets(self):
        prefixes = {'http://www.wikidata.org/entity/': 'wd'}
        result = uri2short('<http://www.wikidata.org/entity/Q5>', prefixes)
        assert result == 'wd:Q5'
    
    def test_non_uri_string(self):
        result = uri2short('Q5', {})
        assert result == 'Q5'
    
    def test_uri_without_matching_prefix(self):
        prefixes = {}
        result = uri2short('http://example.com/resource', prefixes)
        assert result == '<http://example.com/resource>'


class TestReplaceStandardPrefixes:
    def test_replace_uris_with_prefixes(self):
        prefixes = {'http://www.wikidata.org/entity/': 'wd', 'http://www.wikidata.org/prop/direct/': 'wdt'}
        query = 'SELECT * WHERE {<http://www.wikidata.org/entity/Q5> <http://www.wikidata.org/prop/direct/P31> ?o}'
        result = replace_standard_prefixes(query, prefixes)
        assert 'wd:Q5' in result
        assert 'wdt:P31' in result
    
    def test_replace_uris_without_prefixes(self):
        query = "SELECT * WHERE {<http://example.com/resource> ?p ?o}"
        result = replace_standard_prefixes(query, None)
        assert result == query.strip()


class TestRemoveStandardPrefixesDefinition:
    def test_remove_prefix_definitions(self):
        prefixes = {'http://www.wikidata.org/entity/': 'wd'}
        query = 'PREFIX wd: <http://www.wikidata.org/entity/> SELECT * WHERE {?s ?p ?o}'
        result = remove_standard_prefixes_definition(query, prefixes)
        assert 'PREFIX' not in result or 'wd' not in result
    
    def test_keep_non_standard_prefixes(self):
        prefixes = {'http://www.wikidata.org/entity/': 'wd'}
        query = 'PREFIX ex: <http://example.com/> SELECT * WHERE {?s ?p ?o}'
        result = remove_standard_prefixes_definition(query, prefixes)
        assert 'ex:' in result


class TestNormalSparql:
    def test_normalization(self):
        prefixes = {'http://www.wikidata.org/entity/': 'wd'}
        query = 'PREFIX wd: <http://www.wikidata.org/entity/>\n\nSELECT * WHERE {?s ?p ?o}'
        result = normal_sparql(query, prefixes)
        assert result == 'SELECT * WHERE {?s ?p ?o}'


class TestResolveItem:
    def test_variable(self):
        from rdflib.term import Variable
        var = Variable('s')
        result = resolve_item(var)
        assert result == '?s'
    
    def test_dict_with_prefix_and_localname(self):
        entry = {'prefix': 'wd', 'localname': 'Q5'}
        result = resolve_item(entry)
        assert result == 'wd:Q5'
    
    def test_dict_with_mod(self):
        entry = {'prefix': 'wd', 'localname': 'Q5'}
        result = resolve_item(entry, mod='-item')
        assert result == 'wd:Q5-item'


class TestParseQuery:
    def test_parse_simple_select_query(self):
        query = 'SELECT * WHERE {?s ?p ?o}'
        result = parse_query(query)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == ['?s', '?p', '?o']


class TestExtractEntities:
    def test_extract_entities_from_query(self):
        query = 'SELECT * WHERE { wd:Q5 wdt:P31 wd:Q7 }'
        result = extract_entities(query)
        assert 'wd:Q5' in result
        assert 'wdt:P31' in result
        assert 'wd:Q7' in result
    
    def test_extract_entities_with_uri_ref(self):
        parsed = URIRef('http://www.wikidata.org/entity/Q5')
        result = extract_entities_recursive(parsed)
        assert 'http://www.wikidata.org/entity/Q5' in result
