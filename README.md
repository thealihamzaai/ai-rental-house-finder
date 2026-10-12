# AI Rental House Finder

A CrewAI-based rental-property research assistant that combines an LLM with Tavily web search to find source-linked rental options from a user's location, room count, furnishing preference, and budget.

## What it does

- Accepts rental preferences from the command line
- Uses a CrewAI research agent with live Tavily search
- Returns up to five relevant, source-linked rental options
- Avoids claiming unverified prices or availability as facts
- Saves the final research report as Markdown

## Tech stack

- Python
- CrewAI
- OpenAI-compatible LLM via CrewAI
- Tavily Search API
- Requests

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
```

Add your own API keys to `.env`, then run:

```bash
python app.py
```

## Example workflow

A user can ask for a 2-room furnished rental in a city within a specified monthly budget. The agent searches current web sources and produces a concise report with source links.

## Safety and limitations

- Listings, prices, and availability can change quickly.
- Results depend on the quality and freshness of indexed web sources.
- Always verify property details directly with the original listing/provider.
- Never commit API keys or `.env` files to source control.

## Portfolio context

This project demonstrates agentic search, tool use, structured prompting, web-grounded research, and report generation with CrewAI.
