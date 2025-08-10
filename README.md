# D&D Campaign Assistant

An AI-powered D&D campaign assistant that helps you generate and manage content for your D&D campaigns using local LLMs through Ollama.

## Features

- 🎲 Generate NPCs, locations, and scenes with rich detail
- 🔍 Search through your campaign notes and canon
- 📝 Maintain campaign canon and notes
- 🌐 Web UI built with Streamlit
- 🤖 AI-powered content generation using local LLMs through Ollama

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) running locally with your preferred model

## Installation

```bash
pip install dnd-campaign-assistant
```

## Usage

1. Start Ollama with your preferred model:

```bash
ollama serve
ollama pull qwen3:8b  # or your preferred model
```

1. Run the campaign assistant:

```bash
dnd-assistant init  # Initialize data directory
dnd-assistant generate --kind npc --random  # Generate a random NPC
```

1. Or start the web UI:

```bash
streamlit run -m dnd_campaign_assistant.app
```

## Configuration

The assistant can be configured through environment variables:

- `MODEL`: The Ollama model to use (default: qwen3:8b)
- `OLLAMA_HOST`: The Ollama server URL (default: <http://localhost:11434>)
- `OLLAMA_KEY`: Optional API key for Ollama authentication

## Documentation

### CLI Commands

- `init`: Initialize data folder with templates
- `add-note`: Add/Index a session note file
- `search`: Search canon + notes
- `generate`: Generate content (NPCs, locations, scenes)

### Content Generation

The assistant can generate:

- NPCs with personalities, motivations, and story hooks
- Locations with rich descriptions and points of interest
- Scenes with dynamic elements and atmosphere
- Items (coming soon)

### Web UI

The web interface provides:

- Note searching and management
- NPC generation and tracking
- Location creation and management
- Scene generation
- Campaign data overview

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
