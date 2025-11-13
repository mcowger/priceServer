# LiteLLM Model Capabilities Server

A lightweight HTTP server that aggregates and serves large language model (LLM) pricing and capabilities data from multiple providers in the standardized LiteLLM format.

## Overview

This tool fetches model information from various providers and converts them into a unified LiteLLM-compatible format, making it easy to access and compare model capabilities, pricing, and features across different providers through a simple REST API.

### Data Sources

- **OpenRouter API** - Aggregates models from 400+ providers with real-time pricing
- **Models.dev API** - Community-driven database of open and commercial models

## Features

- **Unified API Format**: All models are served in standardized LiteLLM format
- **Multiple Endpoints**: Access all models, specific providers, or filtered data
- **Provider Filtering**: Exclude specific providers from the models.dev source
- **Real-time Data**: Fetches fresh data from upstream APIs on startup
- **Docker Support**: Ready-to-use Docker image available

## Quick Start

### Running with Docker

```bash
docker run -p 8080:8080 mcowger/model-prices-converter
```

The server will be available at `http://localhost:8080`

### Running Locally

```bash
# Install dependencies (Python 3.7+)
pip install -r requirements.txt  # No external dependencies required beyond stdlib

# Start the server
python prices.py --port 8080
```

## API Endpoints

### Get All Models

```bash
GET /models
# or
GET /
```

Returns all models from all providers in LiteLLM format:

```json
{
  "openrouter/openai/gpt-4": {
    "input_cost_per_token": 0.00003,
    "max_input_tokens": 128000,
    "max_output_tokens": 4096,
    "output_cost_per_token": 0.00006,
    "mode": "chat",
    "supported_output_modalities": ["text"],
    "supports_tool_choice": true,
    "litellm_provider": "openrouter",
    ...
  }
}
```

### Get All Providers

```bash
GET /providers
```

Returns metadata about all configured providers:

```json
{
  "openrouter": {
    "id": "openrouter",
    "name": "OpenRouter",
    "models": [...]
  },
  "modelsdev": {
    "id": "modelsdev",
    "name": "Models.dev",
    "models": [...]
  }
}
```

### Get Models from Specific Provider

```bash
GET /provider/{provider_id}
```

Example:
```bash
GET /provider/openrouter
```

Returns only models from the specified provider.

## Configuration

### Command Line Options

```bash
python prices.py [OPTIONS]

Options:
  --port PORT                   Port to run the server on (default: 8080)
  --exclude-modelsdev-provider  Provider to exclude from models.dev source
                                (can be specified multiple times)
```

### Excluding Providers

You can exclude specific providers from the models.dev source using:

**Command Line:**
```bash
python prices.py --exclude-modelsdev-provider openrouter --exclude-modelsdev-provider anthropic
```

**Environment Variable:**
```bash
EXCLUDED_MODELS_DEV_PROVIDERS="openrouter,mistral" python prices.py
```

**Combined:**
```bash
EXCLUDED_MODELS_DEV_PROVIDERS="openrouter" python prices.py --exclude-modelsdev-provider anthropic
```

This is useful for avoiding duplicate models that appear in both OpenRouter and models.dev.

## Model Data Format

Each model includes the following information:

- **id**: Unique model identifier (e.g., "openrouter/openai/gpt-4")
- **name**: Human-readable model name
- **pricing**: Cost per token for input and output
  - `input_cost_per_token`: Cost per input token
  - `output_cost_per_token`: Cost for output token
  - `cache_read_input_token_cost`: Cost for reading cached tokens (if available)
  - `cache_creation_input_token_cost`: Cost for creating cache (if available)
- **limits**: Model constraints
  - `max_input_tokens`: Maximum input context length
  - `max_output_tokens`: Maximum output length
- **capabilities**: Feature support
  - `mode`: Model type (chat, image_generation, audio_speech, embedding)
  - `supported_output_modalities`: Output types (text, image, audio)
  - `supported_modalities`: Input types
  - `supports_tool_choice`: Supports tool selection
  - `supports_function_calling`: Supports function calling
  - `supports_reasoning`: Supports reasoning tokens
  - `supports_vision`: Supports image input
  - `supports_system_messages`: Supports system messages
  - `supports_prompt_caching`: Supports prompt caching
- **litellm_provider**: Provider identifier for LiteLLM compatibility


## Docker Image

A pre-built Docker image is available on Docker Hub:

**Image**: `mcowger/model-prices-converter`

**Registry**: https://hub.docker.com/r/mcowger/model-prices-converter

### Running the Docker Image

```bash
# Basic usage
docker run -p 8080:8080 mcowger/model-prices-converter

# Custom port
docker run -p 9000:8080 mcowger/model-prices-converter --port 8080

# Exclude providers
docker run -p 8080:8080 \
  -e EXCLUDED_MODELS_DEV_PROVIDERS="openrouter,mistral" \
  mcowger/model-prices-converter
```

### Building from Source

```bash
git clone <repository>
cd priceServer
docker build -t model-prices-converter .
docker run -p 8080:8080 model-prices-converter
```

## CI/CD

The project uses GitHub Actions for continuous integration and deployment:

- **Auto-build**: Docker image is automatically built on every commit to `main`
- **Auto-publish**: Built images are published to GitHub Container Registry
- **Multi-tag support**: Images are tagged with branch name, commit SHA, and `latest`

### GitHub Container Registry (Recommended)

A pre-built Docker image is automatically published to GitHub Container Registry:

```bash
# Pull and run the latest image
docker run -p 8080:8080 ghcr.io/mcowger/priceserver

# Use a specific version
docker run -p 8080:8080 ghcr.io/mcowger/priceserver:latest
```

**Registry**: https://ghcr.io/mcowger/priceserver

### Authentication for GitHub Container Registry

```bash
# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u $GITHUB_USERNAME --password-stdin

# Pull the image
docker pull ghcr.io/mcowger/priceserver:latest
```

## Development

### Project Structure

```
priceServer/
├── prices.py          # Main server implementation
├── Dockerfile         # Docker image definition
├── README.md          # This file
└── test_data/         # Sample data files for testing
    ├── openrouter.json
    ├── models.dev.json
    └── ...
```

### Testing with Sample Data

For development or offline use, you can load data from local files:

1. Modify the `OpenRouterProvider` and `ModelsDevProvider` initialization in `run_server()` to use file paths instead of URLs
2. Use the sample data files in `test_data/` directory

## License

This project aggregates data from public APIs. Please respect the terms of service of:
- [OpenRouter](https://openrouter.ai/terms)
- [Models.dev](https://models.dev/)

## Troubleshooting

### Connection Errors

If you see errors fetching from upstream APIs:
- Check internet connectivity
- Verify upstream APIs are accessible
- Use local test data files as fallback

### Port Already in Use

```bash
# Use a different port
python prices.py --port 9000
```

### Memory Usage

The server loads all model data into memory. For large datasets:
- Monitor memory usage with `docker stats`
- Consider increasing Docker memory limits
- Filter out unnecessary providers using exclusion options

## Contributing

Contributions welcome! Areas for improvement:
- Additional data providers
- Response caching for better performance
- Filtering and search capabilities
- Web UI for browsing models

## Support

For issues, questions, or contributions, please refer to the project repository.