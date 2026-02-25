import requests

import pytest
from unittest.mock import patch, MagicMock

from utils.llm import call_LLM


class TestCallLLM:
    @pytest.fixture
    def mock_response(self):
        return {
            'choices': [{'message': {'content': 'Test response'}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 20}
        }

    @pytest.fixture
    def mock_ollama_response(self):
        return {
            'response': 'Test response'
        }

    def test_call_llm_ollama_success(self, mock_response):
        url = 'http://localhost:11434/api/generate'
        key = 'test-key'
        model = 'llama2'
        prompt = 'Test prompt'
        
        with patch('requests.post') as mock_post:
            mock_response_obj = MagicMock()
            mock_response_obj.status_code = 200
            mock_response_obj.json.return_value = mock_response
            mock_post.return_value = mock_response_obj
            
            result = call_LLM(url, key, model, prompt)
            
            assert result == mock_response
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]['json']['model'] == model
            assert call_args[1]['json']['prompt'] == prompt
            assert call_args[1]['headers']['Authorization'] == f'Bearer {key}'

    def test_call_llm_openai_success(self, mock_response):
        url = 'https://api.openai.com/v1/chat/completions'
        key = 'test-key'
        model = 'gpt-4'
        prompt = 'Test prompt'
        
        with patch('requests.post') as mock_post:
            mock_response_obj = MagicMock()
            mock_response_obj.status_code = 200
            mock_response_obj.json.return_value = mock_response
            mock_post.return_value = mock_response_obj
            
            result = call_LLM(url, key, model, prompt)
            
            assert result == mock_response
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]['json']['model'] == model
            assert call_args[1]['json']['messages'][0]['content'] == prompt
            assert call_args[1]['headers']['Authorization'] == f'Bearer {key}'

    def test_call_llm_error_status_code(self):
        url = 'http://localhost:11434/api/generate'
        key = 'test-key'
        model = 'llama2'
        prompt = 'Test prompt'
        
        with patch('requests.post') as mock_post:
            mock_response_obj = MagicMock()
            mock_response_obj.status_code = 400
            mock_response_obj.text = 'Bad Request'
            mock_post.return_value = mock_response_obj
            
            result = call_LLM(url, key, model, prompt)
            
            assert result is None

    def test_call_llm_timeout(self):
        url = 'http://localhost:11434/api/generate'
        key = 'test-key'
        model = 'llama2'
        prompt = 'Test prompt'
        
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout()
            
            result = call_LLM(url, key, model, prompt, timeout=1.0)
            
            assert result is None

    def test_call_llm_connection_error(self):
        url = 'http://localhost:11434/api/generate'
        key = 'test-key'
        model = 'llama2'
        prompt = 'Test prompt'
        
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError()
            
            result = call_LLM(url, key, model, prompt)
            
            assert result is None

    def test_call_llm_keyboard_interrupt(self):
        url = 'http://localhost:11434/api/generate'
        key = 'test-key'
        model = 'llama2'
        prompt = 'Test prompt'
        
        with patch('requests.post') as mock_post:
            mock_post.side_effect = KeyboardInterrupt()
            
            with pytest.raises(KeyboardInterrupt):
                call_LLM(url, key, model, prompt)
