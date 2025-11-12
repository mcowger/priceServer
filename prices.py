#!/usr/bin/env python3
"""
LiteLLM Model Capabilities Server

This script serves information about LLM model capabilities according to the LiteLLM spec.
It defines Model and Provider classes and serves data via HTTP endpoints.

Usage:
    python prices.py [--port PORT]
    
Example:
    python prices.py --port 8080

Endpoints:
    GET http://localhost:8080/models - Get all providers and their models in LiteLLM format
"""

import json
import argparse
import urllib.request
import urllib.error
import logging
from typing import Dict, List, Any, Optional, Union
from http.server import HTTPServer, BaseHTTPRequestHandler
from abc import ABC, abstractmethod
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Output to stdout
    ]
)
logger = logging.getLogger(__name__)


class LiteLLMModel:
    """Represents a LiteLLM model with its capabilities and pricing."""
    
    def __init__(
        self,
        id: str,
        name: str,
        input_cost_per_token: float,
        max_input_tokens: int,
        max_output_tokens: int,
        output_cost_per_token: float,
        mode: str,
        supported_output_modalities: List[str],
        supports_tool_choice: bool,
        litellm_provider: str,
        supported_modalities: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        cache_read_input_token_cost: Optional[float] = None,
        cache_creation_input_token_cost: Optional[float] = None,
        supports_function_calling: Optional[bool] = None,
        supports_reasoning: Optional[bool] = None,
        supports_vision: Optional[bool] = None,
        supports_system_messages: Optional[bool] = None,
        supports_prompt_caching: Optional[bool] = None,
        supports_assistant_prefill: Optional[bool] = None,
        tool_use_system_prompt_tokens: Optional[int] = None,
        **kwargs
    ):
        self.id = id
        self.name = name
        self.input_cost_per_token = input_cost_per_token
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.output_cost_per_token = output_cost_per_token
        self.mode = mode
        self.supported_output_modalities = supported_output_modalities
        self.supports_tool_choice = supports_tool_choice
        self.litellm_provider = litellm_provider
        self.supported_modalities = supported_modalities
        self.max_tokens = max_tokens if max_tokens is not None else max_output_tokens
        self.cache_read_input_token_cost = cache_read_input_token_cost
        self.cache_creation_input_token_cost = cache_creation_input_token_cost
        self.supports_function_calling = supports_function_calling
        self.supports_reasoning = supports_reasoning
        self.supports_vision = supports_vision
        self.supports_system_messages = supports_system_messages
        self.supports_prompt_caching = supports_prompt_caching
        self.supports_assistant_prefill = supports_assistant_prefill
        self.tool_use_system_prompt_tokens = tool_use_system_prompt_tokens
        
        # Store any additional fields
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary representation."""
        result = {
            "input_cost_per_token": self.input_cost_per_token,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "output_cost_per_token": self.output_cost_per_token,
            "mode": self.mode,
            "supported_output_modalities": self.supported_output_modalities,
            "supports_tool_choice": self.supports_tool_choice,
            "litellm_provider": self.litellm_provider
        }
        
        # Add optional fields if they exist
        if self.supported_modalities is not None:
            result["supported_modalities"] = self.supported_modalities
        
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        
        if self.cache_read_input_token_cost is not None:
            result["cache_read_input_token_cost"] = self.cache_read_input_token_cost
        
        if self.cache_creation_input_token_cost is not None:
            result["cache_creation_input_token_cost"] = self.cache_creation_input_token_cost
        
        if self.supports_function_calling is not None:
            result["supports_function_calling"] = self.supports_function_calling
        
        if self.supports_reasoning is not None:
            result["supports_reasoning"] = self.supports_reasoning
        
        if self.supports_vision is not None:
            result["supports_vision"] = self.supports_vision
        
        if self.supports_system_messages is not None:
            result["supports_system_messages"] = self.supports_system_messages
        
        if self.supports_prompt_caching is not None:
            result["supports_prompt_caching"] = self.supports_prompt_caching
        
        if self.supports_assistant_prefill is not None:
            result["supports_assistant_prefill"] = self.supports_assistant_prefill
        
        if self.tool_use_system_prompt_tokens is not None:
            result["tool_use_system_prompt_tokens"] = self.tool_use_system_prompt_tokens
        
        # Add any additional attributes that were passed in kwargs
        for attr_name in dir(self):
            if not attr_name.startswith('_') and attr_name not in [
                'id', 'name', 'input_cost_per_token', 'max_input_tokens', 
                'max_output_tokens', 'output_cost_per_token', 'mode',
                'supported_output_modalities', 'supports_tool_choice', 
                'litellm_provider', 'supported_modalities', 'max_tokens',
                'cache_read_input_token_cost', 'cache_creation_input_token_cost',
                'supports_function_calling', 'supports_reasoning', 
                'supports_vision', 'supports_system_messages',
                'supports_prompt_caching', 'supports_assistant_prefill',
                'tool_use_system_prompt_tokens', 'to_dict'
            ]:
                value = getattr(self, attr_name)
                if not callable(value):
                    result[attr_name] = value
        
        return result


class Provider(ABC):
    """Base class for providers that can provide model information."""
    
    def __init__(self, id: str, name: str):
        self.id = id
        self.name = name
        self.models: List[LiteLLMModel] = []
    
    @abstractmethod
    def update(self) -> None:
        """Update/refresh the provider's data."""
        pass
    
    def get_models(self) -> Dict[str, LiteLLMModel]:
        """Get all models as a dictionary indexed by model ID."""
        return {model.id: model for model in self.models}
    
    def get_model(self, model_id: str) -> Optional[LiteLLMModel]:
        """Get a model by ID from this provider."""
        for model in self.models:
            if model.id == model_id:
                return model
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the provider to a dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "models": [model.to_dict() for model in self.models]
        }


class OpenRouterProvider(Provider):
    """OpenRouter provider that loads data from file or URL and maps to LiteLLM format."""
    
    def __init__(self, file_path: Optional[str] = None, url: Optional[str] = None):
        super().__init__(
            id="openrouter",
            name="OpenRouter"
        )
        self.file_path = file_path
        self.url = url
    
    def update(self) -> None:
        """Load data from file or URL and map to models."""
        logger.info("Loading data...")
        
        raw_data = self._load_raw_data()
        self.models = self._map_data_to_models(raw_data)
        logger.info("Loaded %d models", len(self.models))
    
    def _load_raw_data(self) -> Any:
        """Load raw data from file or URL."""
        if self.file_path:
            file_path = Path(self.file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        elif self.url:
            try:
                headers = {}
                req = urllib.request.Request(self.url, headers=headers)
                with urllib.request.urlopen(req) as response:
                    return json.loads(response.read().decode('utf-8'))
            except urllib.error.URLError as e:
                raise ConnectionError(f"Failed to fetch data from {self.url}: {e}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON received from {self.url}: {e}")
        
        else:
            raise ValueError("Either file_path or url must be specified")
    
    def _map_data_to_models(self, raw_data: Any) -> List[LiteLLMModel]:
        """Map OpenRouter API format to Model objects."""
        models = []
        
        # Check if this is the OpenRouter API response format
        if isinstance(raw_data, dict) and "data" in raw_data:
            models_data = raw_data["data"]
        # Handle both single model and list of models
        else:
            models_data = raw_data if isinstance(raw_data, list) else [raw_data]
        
        for model_data in models_data:
            # Skip if it's not a valid model entry
            if 'id' not in model_data:
                continue
            
            model_id = 'openrouter/' + model_data.get('id')
            if model_id is None:
                continue
            
            # Extract pricing information
            pricing = model_data.get('pricing', {})
            prompt_price = float(pricing.get('prompt', 0))
            completion_price = float(pricing.get('completion', 0))
            cache_read_price = float(pricing.get('input_cache_read', 0))
            
            # Extract architecture and capabilities
            architecture = model_data.get('architecture', {})
            input_modalities = architecture.get('input_modalities', ['text'])
            output_modalities = architecture.get('output_modalities', ['text'])
            
            # Extract context length and limits
            top_provider = model_data.get('top_provider', {})
            context_length = top_provider.get('context_length', 0)
            max_completion_tokens = top_provider.get('max_completion_tokens', 0)
            
            # Determine mode based on modalities and tokenizer
            tokenizer = architecture.get('tokenizer', '')
            mode = "embedding" if "embedding" in tokenizer.lower() else "chat"
            
            # Map OpenRouter-supported parameters to LiteLLM capabilities
            supported_params = model_data.get('supported_parameters', [])
            
            model = LiteLLMModel(
                id=model_id,
                name=model_data.get('name', model_id),
                input_cost_per_token=prompt_price,
                max_input_tokens=context_length,
                max_output_tokens=max_completion_tokens,
                output_cost_per_token=completion_price,
                mode=mode,
                supported_output_modalities=output_modalities,
                supports_tool_choice='tool_choice' in supported_params or 'tools' in supported_params,
                litellm_provider="openrouter",
                supported_modalities=input_modalities,
                cache_read_input_token_cost=cache_read_price,
                supports_reasoning='reasoning' in supported_params,
                supports_function_calling='tools' in supported_params or 'function_calling' in supported_params
            )
            
            models.append(model)
        
        return models


class ModelsDevProvider(Provider):
    """Models.dev provider that loads data from file or URL and maps to LiteLLM format."""
    
    def __init__(self, file_path: Optional[str] = None, url: Optional[str] = None):
        super().__init__(
            id="modelsdev",
            name="Models.dev"
        )
        self.file_path = file_path
        self.url = url
    
    def update(self) -> None:
        """Load data from file or URL and map to models."""
        logger.info("Loading data...")
        
        raw_data = self._load_raw_data()
        self.models = self._map_data_to_models(raw_data)
        logger.info("Loaded %d models", len(self.models))
    
    def _load_raw_data(self) -> Any:
        """Load raw data from file or URL."""
        if self.file_path:
            file_path = Path(self.file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        elif self.url:
            try:
                headers = {}
                req = urllib.request.Request(self.url, headers=headers)
                with urllib.request.urlopen(req) as response:
                    return json.loads(response.read().decode('utf-8'))
            except urllib.error.URLError as e:
                raise ConnectionError(f"Failed to fetch data from {self.url}: {e}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON received from {self.url}: {e}")
        
        else:
            raise ValueError("Either file_path or url must be specified")
    
    def _map_data_to_models(self, raw_data: Any) -> List[LiteLLMModel]:
        """Map models.dev format to Model objects."""
        models = []
        
        # The models.dev format has providers at the top level, each with a "models" object
        if not isinstance(raw_data, dict):
            logger.warning("Expected dictionary from models.dev")
            return models
        
        # Iterate through each provider
        for provider_id, provider_data in raw_data.items():
            if not isinstance(provider_data, dict):
                continue
            
            # Get the models object from the provider
            provider_models = provider_data.get('models', {})
            if not isinstance(provider_models, dict):
                continue
            
            # Iterate through each model in this provider
            for model_key, model_data in provider_models.items():
                if not isinstance(model_data, dict):
                    continue
                
                # The model ID should be the provider ID combined with the model key
                model_id = f"{provider_id}/{model_key}"
                
                # Extract pricing information from cost object
                cost = model_data.get('cost', {})
                prompt_price = float(cost.get('input', 0)) / 1_000_000  # Convert to per-token price
                completion_price = float(cost.get('output', 0)) / 1_000_000  # Convert to per-token price
                cache_read_price = float(cost.get('cache_read', 0)) / 1_000_000 if 'cache_read' in cost else None
                cache_write_price = float(cost.get('cache_write', 0)) / 1_000_000 if 'cache_write' in cost else None
                
                # Extract modalities from modalities object
                modalities = model_data.get('modalities', {})
                input_modalities = modalities.get('input', ['text'])
                output_modalities = modalities.get('output', ['text'])
                
                # Extract limits
                limits = model_data.get('limit', {})
                context_length = limits.get('context', 0)
                max_output_tokens = limits.get('output', 0)
                
                # Determine mode based on output modalities
                if 'image' in output_modalities:
                    mode = "image_generation"
                elif 'audio' in output_modalities:
                    mode = "audio_speech"
                else:
                    mode = "chat"
                
                # Create LiteLLMModel object with correct mappings
                model = LiteLLMModel(
                    id=model_id,
                    name=model_data.get('name', model_id),
                    input_cost_per_token=prompt_price,
                    max_input_tokens=context_length,
                    max_output_tokens=max_output_tokens,
                    output_cost_per_token=completion_price,
                    mode=mode,
                    supported_output_modalities=output_modalities,
                    supports_tool_choice=model_data.get('tool_call', False),
                    litellm_provider="modelsdev",
                    supported_modalities=input_modalities,
                    cache_read_input_token_cost=cache_read_price,
                    cache_creation_input_token_cost=cache_write_price,
                    supports_reasoning=model_data.get('reasoning', False),
                    supports_function_calling=model_data.get('tool_call', False),
                    # Additional fields from the models.dev format
                    knowledge=model_data.get('knowledge', ''),
                    release_date=model_data.get('release_date', ''),
                    last_updated=model_data.get('last_updated', ''),
                    open_weights=model_data.get('open_weights', False)
                )
                
                models.append(model)
        
        return models


class ProviderManager:
    """Manages multiple providers and their data."""
    
    def __init__(self):
        self.providers: List[Provider] = []
    
    def add_provider(self, provider: Provider) -> None:
        """Add a provider to the manager."""
        self.providers.append(provider)
    
    def update_all(self) -> None:
        """Update all providers."""
        logger.info("Updating %d providers...", len(self.providers))
        for provider in self.providers:
            try:
                provider.update()
            except Exception as e:
                logger.error("Provider %s error updating: %s", provider.id, e)
    
    def get_all_models(self) -> Dict[str, LiteLLMModel]:
        """Get all models from all providers."""
        all_models = {}
        for provider in self.providers:
            provider_models = provider.get_models()
            all_models.update(provider_models)
        return all_models
    
    def get_all_providers(self) -> Dict[str, Provider]:
        """Get all providers indexed by provider ID."""
        return {provider.id: provider for provider in self.providers}


class ModelCapabilitiesServer(BaseHTTPRequestHandler):
    """HTTP server for serving LiteLLM model capabilities."""
    
    def __init__(self, provider_manager: ProviderManager, *args, **kwargs):
        self.provider_manager = provider_manager
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/models' or self.path == '/':
            self.serve_models()
        elif self.path == '/providers':
            self.serve_providers()
        elif self.path.startswith('/provider/'):
            provider_id = self.path.split('/', 2)[-1]
            self.serve_provider(provider_id)
        else:
            self.send_error(404, "Not Found")
    
    def serve_models(self):
        """Serve all models in LiteLLM format."""
        try:
            # Convert providers and models to LiteLLM format
            # Format: {"provider/model": {...model_data...}}
            litellm_format = {}
            
            for provider in self.provider_manager.providers:
                for model in provider.models:
                    # Create the combined key: provider_id/model_id
                    combined_key = f"{model.id}"
                    litellm_format[combined_key] = model.to_dict()
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response_json = json.dumps(litellm_format, indent=2)
            self.wfile.write(response_json.encode('utf-8'))
            
            self.log_message(f"Served {len(litellm_format)} models from {len(self.provider_manager.providers)} providers")
            
        except Exception as e:
            self.send_error(500, f"Internal Server Error: {e}")
    
    def serve_providers(self):
        """Serve all providers information."""
        try:
            providers_data = {}
            for provider in self.provider_manager.providers:
                providers_data[provider.id] = provider.to_dict()
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response_json = json.dumps(providers_data, indent=2)
            self.wfile.write(response_json.encode('utf-8'))
            
            self.log_message(f"Served {len(providers_data)} providers")
            
        except Exception as e:
            self.send_error(500, f"Internal Server Error: {e}")
    
    def serve_provider(self, provider_id: str):
        """Serve models from a specific provider."""
        try:
            provider = self.provider_manager.get_all_providers().get(provider_id)
            if not provider:
                self.send_error(404, f"Provider '{provider_id}' not found")
                return
            
            models_data = {model.id: model.to_dict() for model in provider.models}
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response_json = json.dumps(models_data, indent=2)
            self.wfile.write(response_json.encode('utf-8'))
            
            self.log_message(f"Served {len(models_data)} models from provider '{provider_id}'")
            
        except Exception as e:
            self.send_error(500, f"Internal Server Error: {e}")
    
    def log_message(self, format, *args):
        """Override to customize logging."""
        logger.info("Server: " + format, *args)


def run_server(port: int):
    """Run the HTTP server."""
    # Create provider manager
    provider_manager = ProviderManager()
    provider_manager.add_provider(OpenRouterProvider(
        url="https://openrouter.ai/api/v1/models")
    )
    provider_manager.add_provider(ModelsDevProvider(
        url="https://models.dev/api.json")
    )

    # Update all providers to load their data
    provider_manager.update_all()
    
    total_models = sum(len(provider.models) for provider in provider_manager.providers)
    logger.info("Loaded %d providers with %d total models", len(provider_manager.providers), total_models)
    
    # Create the handler with provider manager
    class CustomHandler(ModelCapabilitiesServer):
        def __init__(self, *args, **kwargs):
            super().__init__(provider_manager, *args, **kwargs)
    
    # Create and start the server
    server_address = ('', port)
    httpd = HTTPServer(server_address, CustomHandler)
    
    logger.info("Starting LiteLLM Model Capabilities Server on port %d...", port)
    logger.info("Server endpoints:")
    logger.info("  GET http://localhost:%d/models - Get all models in LiteLLM format", port)
    logger.info("  GET http://localhost:%d/ - Get all models in LiteLLM format", port)
    logger.info("  GET http://localhost:%d/providers - Get all providers", port)
    logger.info("  GET http://localhost:%d/provider/{provider_id} - Get models from specific provider", port)
    logger.info("Press Ctrl+C to stop the server")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nShutting down server...")
        httpd.shutdown()


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Serve LiteLLM model capabilities via HTTP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prices.py --port 8080
  python prices.py
  
Endpoints:
  GET /models - Get all providers and their models in LiteLLM format
  GET / - Get all providers and their models in LiteLLM format
  GET /providers - Get all providers information
  GET /provider/{provider_id} - Get models from a specific provider
        """
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8080,
        help="Port to run the server on (default: 8080)"
    )

    args = parser.parse_args()
    
    run_server(args.port)

if __name__ == "__main__":
    main()
