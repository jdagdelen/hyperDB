# HyperDB

<img src="https://github.com/jdagdelen/hyperDB/blob/main/_static/logo.png?raw=true" width="200" alt="HyperDB Logo">

A hyper-fast local vector database for use with LLM Agents. [Now accepting SAFEs ($35M cap minimum.)](https://www.youtube.com/watch?v=QH2-TGUlwu4)

## Installation

Install the package from PyPI:

```bash
pip install hyperdb-python
```

## Usage

Here's an example of using HyperDB to store and query documents:

```python
import json
from hyperdb import HyperDB

# Load documents from the JSONL file
documents = []

with open("demo/pokemon.jsonl", "r") as f:
    for line in f:
        documents.append(json.loads(line))

# Instantiate HyperDB with the list of documents and the key "description"
db = HyperDB(documents, key="info.description")

# Save the HyperDB instance to a file
db.save("demo/pokemon_hyperdb.json")

# Load the HyperDB instance from the save file
db.load("demo/pokemon_hyperdb.json")

# Query the HyperDB instance with a text input
results = db.query("Likes to sleep.", top_k=5)
```

