import pytest
from utils.text import extract_number, calc_levenshtein_dist, count_sentences


class TestExtractNumber:
    def test_extract_number_valid_qid(self):
        assert extract_number('wd:Q123') == 123
        assert extract_number('wd:Q0') == 0
        assert extract_number('wd:Q999999') == 999999

    def test_extract_number_invalid_format(self):
        assert extract_number('wd:P123') == 0  # Property ID instead of item ID
        assert extract_number('Q123') == 0  # Missing 'wd:' prefix
        assert extract_number('wd:123') == 0  # Missing 'Q' prefix

    def test_extract_number_non_numeric(self):
        assert extract_number('wd:Qabc') == 0
        assert extract_number('wd:Q') == 0

    def test_extract_number_empty_string(self):
        assert extract_number('') == 0

    def test_extract_number_none(self):
        with pytest.raises(AttributeError):
            extract_number(None)


class TestCalcLevenshteinDist:
    def test_identical_strings(self):
        assert calc_levenshtein_dist('hello', 'hello') == 0

    def test_empty_strings(self):
        assert calc_levenshtein_dist('', '') == 0
        assert calc_levenshtein_dist('hello', '') == 5
        assert calc_levenshtein_dist('', 'world') == 5

    def test_completely_different_strings(self):
        assert calc_levenshtein_dist('abc', 'xyz') == 3

    def test_partial_matches(self):
        assert calc_levenshtein_dist('kitten', 'sitting') == 3
        assert calc_levenshtein_dist('flaw', 'lawn') == 2

    def test_case_sensitive(self):
        assert calc_levenshtein_dist('Hello', 'hello') == 1

    def test_single_char_diff(self):
        assert calc_levenshtein_dist('cat', 'bat') == 1
        assert calc_levenshtein_dist('cat', 'ca') == 1


class TestCountSentences:
    def test_single_sentence(self):
        assert count_sentences('Hello world.') == 1
        assert count_sentences('Hello world') == 1  # No period

    def test_multiple_sentences(self):
        assert count_sentences('Hello world. How are you?') == 2
        assert count_sentences('First. Second! Third?') == 3

    def test_empty_string(self):
        assert count_sentences('') == 0

    def test_no_sentences(self):
        assert count_sentences('hello world') == 1  # NLTK treats this as one sentence

    def test_complex_punctuation(self):
        text = 'Dr. Smith went to the store. He bought milk, bread, and eggs. Really!'
        assert count_sentences(text) >= 1

    def test_abbreviations(self):
        # NLTK's sent_tokenize handles common abbreviations
        assert count_sentences('Dr. Smith is here.') == 1

    def test_exception_handling(self):
        # Test with invalid input that might cause exceptions
        assert count_sentences(None) == 0
        assert count_sentences(123) == 0
