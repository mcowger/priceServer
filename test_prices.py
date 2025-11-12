#!/usr/bin/env python3
"""
Pytest tests for the Model and Provider classes functionality.
"""

import pytest
import json
import os
import tempfile
from unittest.mock import Mock, patch
from prices import LiteLLMModel, Provider, SimpleProvider, create_hardcoded_providers, ProviderSource, JSONFileProviderSource, OpenRouterProviderSource


@pytest.fixture
def test_model():
    """Create a test model for testing."""
    return LiteLLMModel(
        id="test-model-1",
        name="Test Model 1",
        input_cost_per_token=1.0e-06,
        max_input_tokens=100000,
        max_output_tokens=8000,
        output_cost_per_token=2.0e-06,
        mode="chat",
        supported_output_modalities=["text"],
        supports_tool_choice=True,
        litellm_provider="test_provider",
        supported_modalities=["text", "image"],
        supports_function_calling=True,
        supports_reasoning=True
    )


@pytest.fixture
def test_provider(test_model):
    """Create a test provider with a model."""
    return Provider(
        id="test_provider",
        name="Test Provider",
        models=[test_model]
    )


@pytest.fixture
def litellm_formatted_providers():
    """Create providers and format them in LiteLLM format."""
    providers = create_hardcoded_providers()
    
    # Convert to LiteLLM format (provider/model as key)
    litellm_format = {}
    
    for provider in providers:
        for model in provider.models:
            combined_key = f"{provider.id}/{model.id}"
            litellm_format[combined_key] = model.to_dict()
    
    return providers, litellm_format


class TestModel:
    """Test cases for the Model class."""
    
    def test_model_creation(self, test_model):
        """Test that a model can be created with all required fields."""
        assert test_model.id == "test-model-1"
        assert test_model.name == "Test Model 1"
        assert test_model.input_cost_per_token == 1.0e-06
        assert test_model.max_input_tokens == 100000
        assert test_model.max_output_tokens == 8000
        assert test_model.output_cost_per_token == 2.0e-06
        assert test_model.mode == "chat"
        assert test_model.supported_output_modalities == ["text"]
        assert test_model.supports_tool_choice is True
        assert test_model.litellm_provider == "test_provider"
    
    def test_model_to_dict(self, test_model):
        """Test that model converts to dictionary correctly."""
        model_dict = test_model.to_dict()
        
        # Check required fields
        required_fields = [
            "input_cost_per_token", "max_input_tokens", "max_output_tokens", 
            "output_cost_per_token", "mode", "supported_output_modalities", 
            "supports_tool_choice", "litellm_provider"
        ]
        
        for field in required_fields:
            assert field in model_dict, f"Missing required field: {field}"
        
        # Check specific values
        assert model_dict["input_cost_per_token"] == 1.0e-06
        assert model_dict["max_input_tokens"] == 100000
        assert model_dict["mode"] == "chat"
        assert model_dict["supports_tool_choice"] is True
    
    def test_model_optional_fields(self, test_model):
        """Test that optional fields are included when present."""
        model_dict = test_model.to_dict()
        
        # Check optional fields
        assert "supported_modalities" in model_dict
        assert model_dict["supported_modalities"] == ["text", "image"]
        assert "supports_function_calling" in model_dict
        assert model_dict["supports_function_calling"] is True
        assert "supports_reasoning" in model_dict
        assert model_dict["supports_reasoning"] is True
    
    def test_model_max_tokens_default(self, test_model):
        """Test that max_tokens defaults to max_output_tokens when not specified."""
        model_dict = test_model.to_dict()
        assert "max_tokens" in model_dict
        assert model_dict["max_tokens"] == test_model.max_output_tokens


class TestProvider:
    """Test cases for the Provider class."""
    
    def test_provider_creation(self, test_provider):
        """Test that a provider can be created with models."""
        assert test_provider.id == "test_provider"
        assert test_provider.name == "Test Provider"
        assert len(test_provider.models) == 1
    
    def test_provider_get_model(self, test_provider):
        """Test that provider can retrieve a model by ID."""
        found_model = test_provider.get_model("test-model-1")
        assert found_model is not None
        assert found_model.id == "test-model-1"
        
        # Test non-existent model
        non_existent = test_provider.get_model("non-existent")
        assert non_existent is None
    
    def test_provider_to_dict(self, test_provider):
        """Test that provider converts to dictionary correctly."""
        provider_dict = test_provider.to_dict()
        
        assert "id" in provider_dict
        assert "name" in provider_dict
        assert "models" in provider_dict
        
        assert provider_dict["id"] == "test_provider"
        assert provider_dict["name"] == "Test Provider"
        assert len(provider_dict["models"]) == 1


class TestLiteLLMFormat:
    """Test cases for LiteLLM format compliance."""
    
    def test_required_fields_present(self, litellm_formatted_providers):
        """Test that all required LiteLLM fields are present in all models."""
        _, litellm_format = litellm_formatted_providers
        
        required_fields = [
            "input_cost_per_token", "max_input_tokens", "max_output_tokens", 
            "output_cost_per_token", "mode", "supported_output_modalities", 
            "supports_tool_choice"
        ]
        
        for model_key, model_data in litellm_format.items():
            for field in required_fields:
                assert field in model_data, f"Model {model_key} missing required field: {field}"
    
    def test_provider_model_key_format(self, litellm_formatted_providers):
        """Test that model keys follow provider/model format."""
        providers, litellm_format = litellm_formatted_providers
        
        expected_providers = {provider.id for provider in providers}
        
        for model_key in litellm_format.keys():
            # Should be in format "provider/model"
            assert "/" in model_key, f"Model key {model_key} should contain '/'"
            
            parts = model_key.split("/")
            assert len(parts) >= 2, f"Model key {model_key} should have at least provider and model parts"
            
            provider_part = parts[0]
            assert provider_part in expected_providers, f"Unknown provider {provider_part} in model key {model_key}"
    
    def test_model_modes(self, litellm_formatted_providers):
        """Test that model modes are valid."""
        _, litellm_format = litellm_formatted_providers
        
        valid_modes = {
            "chat", "embedding", "completion", "image_generation", 
            "audio_transcription", "audio_speech", "moderation", 
            "rerank", "search"
        }
        
        for model_key, model_data in litellm_format.items():
            mode = model_data.get("mode")
            assert mode in valid_modes, f"Model {model_key} has invalid mode: {mode}"
    
    def test_supported_output_modalities(self, litellm_formatted_providers):
        """Test that supported_output_modalities is always present and valid."""
        _, litellm_format = litellm_formatted_providers
        
        for model_key, model_data in litellm_format.items():
            modalities = model_data.get("supported_output_modalities")
            assert modalities is not None, f"Model {model_key} missing supported_output_modalities"
            assert isinstance(modalities, list), f"Model {model_key} supported_output_modalities should be a list"
            assert len(modalities) > 0, f"Model {model_key} supported_output_modalities should not be empty"
    
    def test_litellm_provider_consistency(self, litellm_formatted_providers):
        """Test that litellm_provider is consistent with the key."""
        providers, litellm_format = litellm_formatted_providers
        
        # Create a mapping of provider IDs to expected litellm_providers
        provider_id_to_litellm = {}
        for provider in providers:
            # Map provider ID to expected litellm provider
            # This should match what's in the Model objects
            litellm_providers = {model.litellm_provider for model in provider.models}
            provider_id_to_litellm[provider.id] = litellm_providers
        
        for model_key, model_data in litellm_format.items():
            provider_part = model_key.split("/")[0]
            litellm_provider = model_data.get("litellm_provider")
            
            # The litellm_provider should be in the expected set for this provider
            expected_providers = provider_id_to_litellm.get(provider_part, set())
            assert litellm_provider in expected_providers, \
                f"Model {model_key} litellm_provider {litellm_provider} not in expected set {expected_providers}"


class TestProviderCreation:
    """Test cases for hardcoded provider creation."""
    
    def test_hardcoded_providers_created(self):
        """Test that hardcoded providers are created successfully."""
        providers = create_hardcoded_providers()
        
        assert len(providers) > 0, "Should create at least one provider"
        
        # Check that we have some expected providers
        provider_ids = {provider.id for provider in providers}
        expected_providers = {"openrouter", "anthropic", "openai", "replicate"}
        
        # At least some of the expected providers should be present
        assert len(provider_ids & expected_providers) > 0, \
            f"Expected some of {expected_providers} but got {provider_ids}"
    
    def test_all_providers_have_models(self):
        """Test that all created providers have at least one model."""
        providers = create_hardcoded_providers()
        
        for provider in providers:
            assert len(provider.models) > 0, f"Provider {provider.id} should have at least one model"
    
    def test_total_model_count(self):
        """Test that we get the expected number of models."""
        providers = create_hardcoded_providers()
        
        total_models = sum(len(provider.models) for provider in providers)
        assert total_models == 6, f"Expected 6 models total, got {total_models}"


class TestProviderSource:
    """Test cases for the abstract ProviderSource base class."""
    
    def test_abstract_cannot_instantiate(self):
        """Test that ProviderSource cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ProviderSource()
    
    def test_abstract_methods_must_be_implemented(self):
        """Test that subclasses must implement abstract methods."""
        class IncompleteProviderSource(ProviderSource):
            pass
        
        with pytest.raises(TypeError):
            IncompleteProviderSource()
    
    def test_subclass_implements_abstract_methods(self):
        """Test that a proper subclass can be instantiated."""
        class ConcreteProviderSource(ProviderSource):
            def update(self):
                pass
            
            def get_models(self):
                return {}
        
        # Should not raise an error
        source = ConcreteProviderSource(id="test", name="Test")
        assert source.id == "test"
        assert source.name == "Test"


class TestJSONFileProviderSource:
    """Test cases for the JSONFileProviderSource implementation."""
    
    @pytest.fixture
    def temp_json_file(self):
        """Create a temporary JSON file for testing."""
        test_data = {
            "openai/gpt-4": {
                "input_cost_per_token": 1e-05,
                "max_input_tokens": 128000,
                "max_output_tokens": 4096,
                "output_cost_per_token": 3e-05,
                "mode": "chat",
                "supported_output_modalities": ["text"],
                "supports_tool_choice": True,
                "litellm_provider": "openai"
            },
            "openai/gpt-3.5-turbo": {
                "input_cost_per_token": 5e-07,
                "max_input_tokens": 16385,
                "max_output_tokens": 4096,
                "output_cost_per_token": 1.5e-06,
                "mode": "chat",
                "supported_output_modalities": ["text"],
                "supports_tool_choice": True,
                "litellm_provider": "openai"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    def test_init_with_file_path(self, temp_json_file):
        """Test initialization with file path."""
        source = JSONFileProviderSource(
            id="test-provider",
            name="Test Provider",
            file_path=temp_json_file
        )
        
        assert source.id == "test-provider"
        assert source.name == "Test Provider"
        assert source.file_path == temp_json_file
    
    def test_update_loads_data(self, temp_json_file):
        """Test that update() loads data from file."""
        source = JSONFileProviderSource(
            id="test-provider",
            name="Test Provider",
            file_path=temp_json_file
        )
        
        source.update()
        
        assert source._raw_data is not None
        assert "openai/gpt-4" in source._raw_data
        assert "openai/gpt-3.5-turbo" in source._raw_data
    
    def test_update_file_not_found(self):
        """Test update() with non-existent file."""
        source = JSONFileProviderSource(
            id="test-provider",
            name="Test Provider",
            file_path="/nonexistent/file.json"
        )
        
        with pytest.raises(FileNotFoundError):
            source.update()
    
    def test_get_models_after_update(self, temp_json_file):
        """Test get_models() returns Model objects after update()."""
        source = JSONFileProviderSource(
            id="test-provider",
            name="Test Provider",
            file_path=temp_json_file
        )
        
        source.update()
        models = source.get_models()
        
        assert isinstance(models, dict)
        assert len(models) == 2
        assert "openai/gpt-4" in models
        assert "openai/gpt-3.5-turbo" in models
        
        # Check first model
        model = models["openai/gpt-4"]
        assert isinstance(model, LiteLLMModel)
        assert model.id == "openai/gpt-4"
        assert model.input_cost_per_token == 1e-05
        assert model.max_input_tokens == 128000
        assert model.output_cost_per_token == 3e-05
        assert model.litellm_provider == "openai"
    
    def test_get_models_before_update(self, temp_json_file):
        """Test get_models() before calling update()."""
        source = JSONFileProviderSource(
            id="test-provider",
            name="Test Provider",
            file_path=temp_json_file
        )
        
        models = source.get_models()
        assert models == {}
    
    def test_to_provider(self, temp_json_file):
        """Test to_provider() method."""
        source = JSONFileProviderSource(
            id="openai",
            name="OpenAI",
            file_path=temp_json_file
        )
        
        source.update()
        provider = source.to_provider()
        
        assert isinstance(provider, SimpleProvider)
        assert provider.id == "openai"
        assert provider.name == "OpenAI"
        assert len(provider.models) == 2
        
        # Check that models are accessible
        model = provider.get_model("openai/gpt-4")
        assert model is not None
        assert model.id == "openai/gpt-4"
    
    def test_empty_json_file(self):
        """Test with empty JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)
            temp_path = f.name
        
        try:
            source = JSONFileProviderSource(
                id="test-provider",
                name="Test Provider",
                file_path=temp_path
            )
            
            source.update()
            models = source.get_models()
            assert models == {}
            
            provider = source.to_provider()
            assert len(provider.models) == 0
            
        finally:
            os.unlink(temp_path)


class TestOpenRouterProviderSource:
    """Test cases for the OpenRouterProviderSource implementation."""
    
    @pytest.fixture
    def openrouter_test_data(self):
        """Fixture with OpenRouter API response data."""
        return {
            "data": [
                {
                    "id": "openai/gpt-4-turbo",
                    "canonical_slug": "openai/gpt-4-turbo",
                    "name": "OpenAI: GPT-4 Turbo",
                    "context_length": 128000,
                    "pricing": {
                        "prompt": "0.00001",
                        "completion": "0.00003",
                        "request": "0",
                        "image": "0",
                        "web_search": "0",
                        "internal_reasoning": "0",
                        "input_cache_read": "0.0000001"
                    },
                    "architecture": {
                        "modality": "text->text",
                        "input_modalities": ["text"],
                        "output_modalities": ["text"]
                    },
                    "top_provider": {
                        "context_length": 128000,
                        "max_completion_tokens": 4096,
                        "is_moderated": False
                    },
                    "supported_parameters": [
                        "max_tokens",
                        "temperature",
                        "top_p"
                    ]
                },
                {
                    "id": "anthropic/claude-3-opus",
                    "canonical_slug": "anthropic/claude-3-opus",
                    "name": "Anthropic: Claude 3 Opus",
                    "context_length": 200000,
                    "pricing": {
                        "prompt": "0.000015",
                        "completion": "0.000075",
                        "request": "0",
                        "image": "0",
                        "web_search": "0",
                        "internal_reasoning": "0",
                        "input_cache_read": "0.00000015"
                    },
                    "architecture": {
                        "modality": "text+image->text",
                        "input_modalities": ["text", "image"],
                        "output_modalities": ["text"]
                    },
                    "top_provider": {
                        "context_length": 200000,
                        "max_completion_tokens": 4096,
                        "is_moderated": False
                    },
                    "supported_parameters": [
                        "max_tokens",
                        "temperature",
                        "top_p"
                    ]
                }
            ]
        }
    
    def test_init(self):
        """Test initialization."""
        source = OpenRouterProviderSource(
            id="openrouter",
            name="OpenRouter"
        )
        
        assert source.id == "openrouter"
        assert source.name == "OpenRouter"
        assert source.api_url == "https://openrouter.ai/api/v1/models"
    
    @patch('urllib.request.urlopen')
    def test_update_with_mock_response(self, mock_urlopen, openrouter_test_data):
        """Test update() with mocked API response."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(openrouter_test_data).encode('utf-8')
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_response
        
        source = OpenRouterProviderSource(
            id="openrouter",
            name="OpenRouter"
        )
        
        source.update()
        
        # Check that urlopen was called
        assert mock_urlopen.called
        
        assert source._raw_data == openrouter_test_data
    
    @patch('urllib.request.urlopen')
    def test_get_models_after_update(self, mock_urlopen, openrouter_test_data):
        """Test get_models() after update() with mocked response."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(openrouter_test_data).encode('utf-8')
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_response
        
        source = OpenRouterProviderSource(
            id="openrouter",
            name="OpenRouter"
        )
        
        source.update()
        models = source.get_models()
        
        assert len(models) == 2
        assert "openai/gpt-4-turbo" in models
        assert "anthropic/claude-3-opus" in models
        
        # Check first model
        model1 = models["openai/gpt-4-turbo"]
        assert isinstance(model1, LiteLLMModel)
        assert model1.id == "openai/gpt-4-turbo"
        assert model1.input_cost_per_token == 1e-05
        assert model1.max_input_tokens == 128000
        assert model1.max_output_tokens == 4096
        assert model1.output_cost_per_token == 3e-05
        assert model1.mode == "chat"
        assert model1.supported_output_modalities == ["text"]
        assert model1.supported_modalities == ["text"]
        assert model1.supports_tool_choice == True
        assert model1.litellm_provider == "openrouter"
        assert model1.cache_read_input_token_cost == 1e-07
        
        # Check second model (multimodal)
        model2 = models["anthropic/claude-3-opus"]
        assert isinstance(model2, LiteLLMModel)
        assert model2.id == "anthropic/claude-3-opus"
        assert model2.supported_modalities == ["text", "image"]
        assert model2.max_input_tokens == 200000
        assert model2.max_output_tokens == 4096
    
    @patch('prices.requests.get')
    def test_to_provider(self, mock_get, openrouter_test_data):
        """Test to_provider() method."""
        mock_response = Mock()
        mock_response.json.return_value = openrouter_test_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        source = OpenRouterProviderSource(
            id="openrouter",
            name="OpenRouter"
        )
        
        source.update()
        provider = source.to_provider()
        
        assert isinstance(provider, SimpleProvider)
        assert provider.id == "openrouter"
        assert provider.name == "OpenRouter"
        assert len(provider.models) == 2
    
    @patch('prices.requests.get')
    def test_get_models_before_update(self, mock_get, openrouter_test_data):
        """Test get_models() before update()."""
        mock_response = Mock()
        mock_response.json.return_value = openrouter_test_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        source = OpenRouterProviderSource(
            id="openrouter",
            name="OpenRouter"
        )
        
        models = source.get_models()
        assert models == {}
    
    def test_convert_openrouter_model_basic(self, openrouter_test_data):
        """Test _convert_openrouter_model with basic text-only model."""
        source = OpenRouterProviderSource(
            id="openrouter",
            name="OpenRouter"
        )
        
        openrouter_model = openrouter_test_data["data"][0]  # GPT-4 Turbo
        model = source._convert_openrouter_model(openrouter_model)
        
        assert isinstance(model, LiteLLMModel)
        assert model.id == "openai/gpt-4-turbo"
        assert model.input_cost_per_token == 1e-05
        assert model.max_input_tokens == 128000
        assert model.max_output_tokens == 4096
        assert model.output_cost_per_token == 3e-05
        assert model.mode == "chat"
        assert model.supported_output_modalities == ["text"]
        assert model.supported_modalities == ["text"]
        assert model.supports_tool_choice == True
        assert model.litellm_provider == "openrouter"
        assert model.cache_read_input_token_cost == 1e-07
    
    def test_convert_openrouter_model_multimodal(self, openrouter_test_data):
        """Test _convert_openrouter_model with multimodal model."""
        source = OpenRouterProviderSource(
            id="openrouter",
            name="OpenRouter"
        )
        
        openrouter_model = openrouter_test_data["data"][1]  # Claude 3 Opus
        model = source._convert_openrouter_model(openrouter_model)
        
        assert isinstance(model, LiteLLMModel)
        assert model.id == "anthropic/claude-3-opus"
        assert model.supported_modalities == ["text", "image"]
        assert model.max_input_tokens == 200000
        assert model.max_output_tokens == 4096
    
    @patch('prices.requests.get')
    def test_handles_api_error(self, mock_get):
        """Test error handling when API call fails."""
        mock_get.side_effect = Exception("API Error")
        
        source = OpenRouterProviderSource(
            id="openrouter",
            name="OpenRouter"
        )
        
        with pytest.raises(Exception):
            source.update()


class TestIntegration:
    """Test integration between ProviderSource classes and existing Model/Provider classes."""
    
    def test_provider_source_to_provider_creates_valid_provider(self):
        """Test that ProviderSource creates valid Provider objects."""
        test_data = {
            "openai/gpt-4": {
                "input_cost_per_token": 1e-05,
                "max_input_tokens": 128000,
                "max_output_tokens": 4096,
                "output_cost_per_token": 3e-05,
                "mode": "chat",
                "supported_output_modalities": ["text"],
                "supports_tool_choice": True,
                "litellm_provider": "openai",
                "supported_modalities": ["text", "image"],
                "supports_reasoning": True
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            source = JSONFileProviderSource(
                id="test-provider",
                name="Test Provider",
                file_path=temp_path
            )
            
            source.update()
            provider = source.to_provider()
            
            # Test Provider methods
            assert provider.id == "test-provider"
            assert provider.name == "Test Provider"
            
            # Test get_model method
            model = provider.get_model("openai/gpt-4")
            assert model is not None
            assert model.id == "openai/gpt-4"
            
            # Test to_dict method
            provider_dict = provider.to_dict()
            assert "id" in provider_dict
            assert "name" in provider_dict
            assert "models" in provider_dict
            assert len(provider_dict["models"]) == 1
            
            # Test model dict contains all expected fields
            model_dict = provider_dict["models"][0]
            assert model_dict["id"] == "openai/gpt-4"
            assert model_dict["input_cost_per_token"] == 1e-05
            assert model_dict["mode"] == "chat"
            assert model_dict["supported_modalities"] == ["text", "image"]
            
        finally:
            os.unlink(temp_path)
    
    @patch('prices.requests.get')
    def test_openrouter_to_provider_integration(self, mock_get):
        """Test integration of OpenRouterProviderSource with Provider class."""
        test_data = {
            "data": [
                {
                    "id": "openai/gpt-5-codex",
                    "canonical_slug": "openai/gpt-5-codex",
                    "name": "OpenAI: GPT-5 Codex",
                    "context_length": 400000,
                    "pricing": {
                        "prompt": "0.00000125",
                        "completion": "0.00001",
                        "request": "0",
                        "image": "0",
                        "web_search": "0",
                        "internal_reasoning": "0",
                        "input_cache_read": "0.000000125"
                    },
                    "architecture": {
                        "modality": "text+image->text",
                        "input_modalities": ["text", "image"],
                        "output_modalities": ["text"]
                    },
                    "top_provider": {
                        "context_length": 400000,
                        "max_completion_tokens": 128000,
                        "is_moderated": True
                    },
                    "supported_parameters": [
                        "include_reasoning",
                        "max_tokens",
                        "reasoning",
                        "response_format",
                        "seed",
                        "structured_outputs",
                        "tool_choice",
                        "tools"
                    ]
                }
            ]
        }
        
        mock_response = Mock()
        mock_response.json.return_value = test_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        source = OpenRouterProviderSource(
            id="openrouter",
            name="OpenRouter"
        )
        
        source.update()
        provider = source.to_provider()
        
        # Test provider structure
        assert provider.id == "openrouter"
        assert provider.name == "OpenRouter"
        assert len(provider.models) == 1
        
        # Test model structure
        model = provider.models[0]
        assert model.id == "openai/gpt-5-codex"
        assert model.name == "OpenAI: GPT-5 Codex"
        assert model.input_cost_per_token == 1.25e-06
        assert model.max_input_tokens == 400000
        assert model.max_output_tokens == 128000
        assert model.output_cost_per_token == 1e-05
        assert model.mode == "chat"
        assert model.supported_modalities == ["text", "image"]
        assert model.supports_tool_choice == True
        assert model.litellm_provider == "openrouter"
        assert model.cache_read_input_token_cost == 1.25e-07
        assert model.supports_reasoning == True