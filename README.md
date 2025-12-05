
A FastAPI application for transforming questions and SPARQL queries over Wikidata by replacing entities with semantically similar alternatives.

### Features

- **Entity Substitution**: Finds and replaces entities in Wikidata queries while maintaining query validity
- **Multi-language Support**: Handles questions in English, German, French, Russian, and Ukrainian
- **Complexity Levels**: Generates variations at different difficulty levels (easy, normal, hard, random)
- **LLM Integration**: Uses an LLM to rephrase questions after entity replacement
- **PageRank Scoring**: Ranks substitutes based on entity popularity
- **MongoDB Caching**: Caches SPARQL results for performance
- **REST API**: Exposes `/transform` endpoint for batch processing

### Requirements

- FastAPI, Pydantic
- PyMongo
- NLTK
- requests
- python-decouple
- dynutils module
- MongoDB instance (local or via Docker)
- uvicorn

### Environment Variables

Required in `.env`:
- `MONGO_HOST`, `MONGO_USER`, `MONGO_PASS`
- `LLM_URL`, `KEY` (LLM service credentials, empty string for local instance)
- `WIKIDATA_ENDPOINT`, `WIKIDATA_AGENT`

### Usage

**CLI:**
```bash
python3 dynbench.py --query "SELECT ?answer WHERE { wd:Q14452 wdt:P17 ?answer }" --question "Which country does the famous Easter island belong to?" --language en --complexity normal --model "mistral-small:latest"
```

**uvicorn:**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Docker


**Build:**
```bash
docker build -t dynbench .
```

**Run:**
```bash
docker run --add-host=host.docker.internal:host-gateway -e MONGO_URL="mongodb://host.docker.internal:27017" -p 8000:8000 dynbench
```

The API will be available at `http://localhost:8000`.