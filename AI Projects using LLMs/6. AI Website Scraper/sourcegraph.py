from scrapegraphai.graphs import SmartScraperGraph

graph_config = {
    # "llm": {
    #     "model": "ollama/mistral",
    #     "temperature": 0,
    #     "format": "json",  # Ollama needs the format to be specified explicitly
    #     "base_url": "http://localhost:11434",  # set Ollama URL
    # },
    "llm": {
        "model": "groq/llama3-70b-8192",
        "api_key": "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",
        "temperature": 0
    },
    "embeddings": {
        "model": "ollama/nomic-embed-text",
        "base_url": "http://localhost:11434",  # set Ollama URL
    }
}
print("Configuration Completed")
smart_scraper_graph = SmartScraperGraph(
    prompt="What are the teams fighting for last two spots in playoffs",
    # also accepts a string with the already downloaded HTML code
    source="https://www.espncricinfo.com/",
    config=graph_config
)
print("Scraper Created")

import nest_asyncio
nest_asyncio.apply()

result = smart_scraper_graph.run()
print(result)