import os
import re
from datetime import datetime
from pathlib import Path

import requests
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")
if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY is missing. Add it to your .env file.")

llm = LLM(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0.3)


@tool("Tavily Rental Search")
def tavily_search_rent(query: str) -> str:
    """Search the web for rental-property information using Tavily."""
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": 5,
        "include_answer": True,
        "search_depth": "advanced",
    }
    try:
        response = requests.post("https://api.tavily.com/search", json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return f"Search unavailable: {exc}"

    lines = []
    if data.get("answer"):
        lines.append(f"Summary: {data['answer']}")

    for idx, result in enumerate(data.get("results", [])[:5], 1):
        lines.append(
            f"{idx}. {result.get('title', 'Untitled')}\n"
            f"   {result.get('content', '')[:350]}\n"
            f"   Source: {result.get('url', 'N/A')}"
        )
    return "\n\n".join(lines) if lines else "No relevant rental results found."


rental_agent = Agent(
    role="Rental Property Research Agent",
    goal="Find relevant rental-property options that match the user's stated preferences.",
    backstory=(
        "You are a careful rental-market research assistant. You use live web search, "
        "preserve source URLs, and avoid presenting availability or pricing as guaranteed."
    ),
    tools=[tavily_search_rent],
    llm=llm,
    verbose=False,
    allow_delegation=False,
)


def safe_slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return value.strip("_") or "search"


def search_rentals(city: str, rooms: int, furnished: str, budget: float, currency: str = "USD") -> dict:
    furnished_text = "furnished" if furnished.lower() in {"yes", "y", "true"} else "unfurnished"
    query = (
        f"rental apartments or houses in {city}, {rooms} rooms, {furnished_text}, "
        f"maximum monthly budget {budget:g} {currency}; include current listing links where available"
    )

    task = Task(
        description=(
            f"Research rental options for this request: {query}.\n"
            "Return up to five useful options. For each option include property/location, price if found, "
            "key details, and the source URL. Clearly label information that could not be verified. "
            "Do not invent listings, prices, addresses, or availability."
        ),
        expected_output="A concise, source-linked rental research report with up to five relevant options.",
        agent=rental_agent,
    )

    crew = Crew(agents=[rental_agent], tasks=[task], process=Process.sequential, verbose=False)
    crew.kickoff()
    output = getattr(task.output, "raw", str(task.output))

    return {
        "search_details": {
            "city": city,
            "rooms": rooms,
            "furnished": furnished_text,
            "budget": f"{budget:g} {currency}",
        },
        "rentals": output,
    }


def save_report(result: dict, output_dir: str = "outputs") -> Path:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    details = result["search_details"]
    output_path = Path(output_dir) / f"rental_report_{safe_slug(details['city'])}.md"
    content = f"""# Rental Property Research Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Search criteria

- **City:** {details['city']}
- **Rooms:** {details['rooms']}
- **Furnished:** {details['furnished']}
- **Budget:** {details['budget']}

## Results

{result['rentals']}

## Important note

Rental listings, prices, and availability can change quickly. Verify all details directly with the source or property provider before making a decision or payment.
"""
    output_path.write_text(content, encoding="utf-8")
    return output_path


def main() -> None:
    print("\nAI Rental House Finder\n")
    city = input("City: ").strip() or "Karachi, Pakistan"
    rooms = int(input("Number of rooms: ").strip() or "2")
    furnished = input("Furnished? (yes/no): ").strip().lower() or "no"
    budget = float(input("Maximum monthly budget: ").strip() or "1200")
    currency = input("Currency (e.g. USD, PKR): ").strip().upper() or "USD"

    result = search_rentals(city, rooms, furnished, budget, currency)
    print("\n" + result["rentals"])
    path = save_report(result)
    print(f"\nReport saved to: {path}")


if __name__ == "__main__":
    main()
