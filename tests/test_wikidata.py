import pytest
from unittest.mock import Mock, patch
import json

from utils.wikidata import (
    query_wikidata_label,
    get_wikidata_label,
    get_resources_types,
    find_substitutes,
    check_productivity_single,
    WIKIDATA_PREFIX,
    FIXED_LABELS
)


class TestQueryWikidataLabel:
    """Tests for query_wikidata_label function."""

    def test_with_fixed_labels(self):
        """Test that fixed labels are used when provided."""
        uri = 'http://www.w3.org/2000/01/rdf-schema#label'
        fixed_labels = {'http://www.w3.org/2000/01/rdf-schema#label': 'rdfs:label'}
        
        result = query_wikidata_label(uri, Mock(), fixed_labels=fixed_labels)
        
        assert result == 'rdfs:label'

    def test_with_none_fixed_labels(self):
        """Test that None fixed_labels doesn't cause issues."""
        uri = 'wd:Q123'
        
        mock_execute = Mock(return_value={
            'results': {'bindings': [{'label': {'value': 'Test Label'}}]}
        })
        
        result = query_wikidata_label(uri, mock_execute, fixed_labels=None)
        
        assert result == 'Test Label'

    def test_successful_label_query(self):
        """Test successful label query with English language."""
        uri = 'wd:Q123'

        mock_execute = Mock(return_value = {
            'results': {
                'bindings': [{'label': {'type': 'literal', 'value': 'Test Entity'}}]
            }
        })
        
        result = query_wikidata_label(uri, mock_execute)
        
        assert result == 'Test Entity'

    def test_label_query_with_default_language(self):
        """Test label query falls back to default language."""
        uri = 'wd:Q123'
        
        mock_execute = Mock(return_value = {
            'results': {
                'bindings': [{'label': {'value': 'Default Label'}}]
            }
        })
        
        result = query_wikidata_label(uri, mock_execute, lang='fr')
        
        assert result == 'Default Label'

    def test_label_query_with_redirect(self):
        """Test label query with redirect handling."""
        uri = 'wd:Q123'
        
        mock_execute = Mock(return_value = {
            'results': {
                'bindings': [{'label': {'value': 'Redirected Label'}}]
            }
        })
        
        result = query_wikidata_label(uri, mock_execute)
        
        assert result == 'Redirected Label'

    def test_label_query_no_results(self):
        """Test label query with no results."""
        uri = 'wd:Q123'
        
        mock_execute = Mock(return_value = {'results': {'bindings': []}})
        
        result = query_wikidata_label(uri, mock_execute)
        
        assert result is None

    def test_label_query_exception_handling(self):
        """Test exception handling in label query."""
        uri = 'wd:Q123'
        
        mock_execute = Mock(error=True)

        result = query_wikidata_label(uri, mock_execute)
        
        assert result is None

    def test_label_query_keyboard_interrupt(self):
        """Test that KeyboardInterrupt is re-raised."""
        uri = 'wd:Q123'
        mock_execute = Mock(side_effect=KeyboardInterrupt())
        
        with pytest.raises(KeyboardInterrupt):
            query_wikidata_label(uri, mock_execute)


class TestGetWikidataLabel:
    """Tests for get_wikidata_label function."""

    def test_successful_label_with_prefix(self):
        """Test getting label with prefix conversion."""
        uri = 'http://www.wikidata.org/entity/Q123'
        
        mock_execute = Mock(return_value = {
            'results': {
                'bindings': [{'label': {'value': 'Test Entity'}}]
            }
        })
        
        result = get_wikidata_label(uri, mock_execute)
        
        assert result == 'Test Entity'

    def test_get_label_with_custom_prefixes(self):
        """Test getting label with custom prefixes."""
        uri = 'http://www.wikidata.org/entity/Q456'
        custom_prefixes = WIKIDATA_PREFIX.copy()
        
        mock_execute = Mock(return_value = {
            'results': {
                'bindings': [{'label': {'value': 'Custom Prefix Entity'}}]
            }
        })
        
        result = get_wikidata_label(uri, mock_execute)
        
        assert result == 'Custom Prefix Entity'

    def test_get_label_exception_handling(self):
        """Test exception handling in get_wikidata_label."""
        uri = 'http://www.wikidata.org/entity/Q123'
        
        mock_execute = Mock(error=True)

        result = get_wikidata_label(uri, mock_execute)
        
        assert result is None


class TestGetResourcesTypes:
    """Tests for get_resources_types function."""

    def test_empty_predicates(self):
        """Test with empty predicates list."""
        info = {'resources': ['wd:Q123']}
        result = get_resources_types(info, Mock(), predicates=[])
        
        assert result == {}

    def test_non_wikidata_resources(self):
        """Test filtering of non-Wikidata resources."""
        info = {'resources': ['wd:Q123', 'dbpedia:Res1', 'wd:Q456']}
        predicates = ['wdt:P31', 'wdt:P279']
        
        mock_execute = Mock(return_value={
            'results': {
                'bindings': [
                    {'p': {'type': 'uri', 'value': 'wdt:P31'}, 'o': {'type': 'uri', 'value': 'wd:Q1'}},
                    {'p': {'type': 'uri', 'value': 'wdt:P279'}, 'o': {'type': 'uri', 'value': 'wd:Q2'}}
                ]
            }
        })
        
        result = get_resources_types(info, mock_execute, predicates=predicates)
        
        assert 'wd:Q123' in result
        assert 'dbpedia:Res1' not in result
        assert 'wd:Q456' in result

    def test_successful_type_collection(self):
        """Test successful type collection."""
        info = {'resources': ['wd:Q123']}
        predicates = ['wdt:P31', 'wdt:P279']
        
        mock_execute = Mock(return_value={
            'results': {
                'bindings': [
                    {'p': {'type': 'uri', 'value': 'wdt:Q123'}, 'o': {'type': 'uri', 'value': 'wd:Q1'}},
                    {'p': {'type': 'uri', 'value': 'wdt:Q456'}, 'o': {'type': 'uri', 'value': 'wd:Q2'}}
                ]
            }
        })
        
        result = get_resources_types(info, mock_execute, predicates=predicates)
        
        assert 'wd:Q123' in result
        assert 'wdt:P31' in result['wd:Q123']
        assert 'wdt:P279' in result['wd:Q123']

    def test_type_collection_exception_handling(self):
        """Test exception handling in type collection."""
        info = {'resources': ['wd:Q123']}
        predicates = ['wdt:P31']
        
        mock_execute = Mock(side_effect = Exception("SPARQL error"))
        
        result = get_resources_types(info, mock_execute, predicates=predicates)
        
        assert result == {'wd:Q123': {}}

    def test_type_collection_keyboard_interrupt(self):
        """Test that KeyboardInterrupt is re-raised."""
        info = {'resources': ['wd:Q123']}
        predicates = ['wdt:P31']
        mock_execute = Mock(side_effect=KeyboardInterrupt())
        
        with pytest.raises(KeyboardInterrupt):
            get_resources_types(info, mock_execute, predicates=predicates)


class TestFindSubstitutes:
    """Tests for find_substitutes function."""

    def test_empty_entities(self):
        """Test with no Wikidata entities."""
        query = "SELECT ?x WHERE { ?x ?p ?o }"
        info = {'resources': ['dbpedia:Res1'], 'conditions': {}, 'query conditions': {}}
        
        result = find_substitutes(query, Mock(), info)
        
        assert result == []

    def test_successful_substitute_finding(self):
        """Test successful substitute finding."""
        query = "SELECT ?x WHERE { ?x ?p ?o }"
        info = {
            'resources': ['wd:Q123'],
            'conditions': {
                'wd:Q123': {
                    'wdt:P31': ['wd:Q1'],
                    'wdt:P279': ['wd:Q2']
                }
            },
            'query conditions': {
                'wd:Q123': ['wdt:P31']
            }
        }
        
        mock_execute = Mock(return_value = {
            'results': {
                'bindings': [
                    {
                        'subst': {'type': 'uri', 'value': 'wd:Q456'},
                        'label': {'type': 'literal', 'value': 'Substitute'},
                        'lang': {'type': 'literal', 'value': 'en'}
                    }
                ]
            }
        })
        
        result = find_substitutes(query, mock_execute, info)
        
        assert len(result) == 1
        assert result[0]['entity'] == 'wd:Q123'
        assert len(result[0]['results']) == 1
        assert result[0]['results'][0]['old'] == 'wd:Q123'

    def test_substitute_finding_exception_handling(self):
        """Test exception handling in substitute finding."""
        query = "SELECT ?x WHERE { ?x ?p ?o }"
        info = {
            'resources': ['wd:Q123'],
            'conditions': {
                'wd:Q123': {
                    'wdt:P31': [],
                    'wdt:P279': []
                }
            },
            'query conditions': {
                'wd:Q123': []
            }
        }
        
        mock_execute = Mock(side_effect = Exception("SPARQL error"))
        
        result = find_substitutes(query, mock_execute, info)
        
        assert result == []

    def test_substitute_finding_keyboard_interrupt(self):
        """Test that KeyboardInterrupt is re-raised."""
        query = "SELECT ?x WHERE { ?x ?p ?o }"
        info = {
            'resources': ['wd:Q123'],
            'conditions': {
                'wd:Q123': {
                    'wdt:P31': [],
                    'wdt:P279': []
                }
            },
            'query conditions': {
                'wd:Q123': []
            }
        }
        mock_execute = Mock(side_effect=KeyboardInterrupt())
        
        with pytest.raises(KeyboardInterrupt):
            find_substitutes(query, mock_execute, info)


class TestCheckProductivitySingle:
    """Tests for check_productivity_single function."""

    def test_productive_query(self):
        """Test with productive query returning results."""
        query = "SELECT ?x WHERE { ?x ?p wd:Q123 }"
        replace = {'old': 'wd:Q123', 'new': 'wd:Q456'}
        
        mock_execute = Mock(return_value = {
            'results': {
                'bindings': [{'?x': {'type': 'literal', 'value': 'test'}}]
            }
        })
        
        result = check_productivity_single(query, mock_execute, replace)
        
        assert result is True

    def test_non_productive_query(self):
        """Test with non-productive query returning no results."""
        query = "SELECT ?x WHERE { ?x ?p ?o }"
        replace = {'old': '?x', 'new': 'wd:Q123'}
        
        mock_execute = Mock(return_value = {'results': {'bindings': []}})
        
        result = check_productivity_single(query, mock_execute, replace)
        
        assert result is False

    def test_query_with_default_replace(self):
        """Test query with default replace dictionary."""
        query = "SELECT ?x WHERE { ?x ?p ?o }"
        
        mock_execute = Mock(return_value = { 'results': {'bindings': []} })
        
        result = check_productivity_single(query, mock_execute, {})
        
        assert result is False

    def test_productivity_check_exception_handling(self):
        """Test exception handling in productivity check."""
        query = "SELECT ?x WHERE { ?x ?p ?o }"
        replace = {'old': '?x', 'new': 'wd:Q123'}
        
        mock_execute = Mock(side_effect = Exception("SPARQL error"))
        
        result = check_productivity_single(query, mock_execute, replace)
        
        assert result is False

    def test_custom_prefixes(self):
        """Test with custom prefixes."""
        query = "SELECT ?x WHERE { ?x ?p ?o }"
        replace = {'old': '?x', 'new': 'wd:Q123'}
        custom_prefixes = {'custom': 'http://custom.namespace/'}
        
        mock_execute = Mock(return_value = { 'results': { 'bindings': [] } })
        
        result = check_productivity_single(query, mock_execute, replace, custom_prefixes)
        
        assert result is False
