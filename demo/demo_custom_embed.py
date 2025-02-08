import json
from hyperdb import HyperDB
# To use this, install sentence-transformers
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer


# Load documents from the JSONL file
documents = []

with open("demo/pokemon.jsonl", "r") as f:
    for line in f:
        documents.append(json.loads(line))

# Instantiate HyperDB with the list of documents and the key "description"
model = SentenceTransformer('all-MiniLM-L6-v2')
db = HyperDB(documents, key="info.description",
             embedding_function=model.encode)

# Save the HyperDB instance to a file
db.save("demo/pokemon_hyperdb.pickle.gz")

# Load the HyperDB instance from the file
db.load("demo/pokemon_hyperdb.pickle.gz")

# Query the HyperDB instance with a text input
results = db.query("Likes to sleep.", top_k=5)

# Define a function to pretty print the results


def format_entry(pokemon):
    name = pokemon["name"]
    hp = pokemon["hp"]
    info = pokemon["info"]
    pokedex_id = info["id"]
    pkm_type = info["type"]
    weakness = info["weakness"]
    description = info["description"]

    pretty_pokemon = f"""Name: {name}
Pokedex ID: {pokedex_id}
HP: {hp}
Type: {pkm_type}
Weakness: {weakness}
Description: {description}
"""
    return pretty_pokemon


# Print the top 5 most similar Pokémon descriptions
for pokemon, _ in results:
    print(format_entry(pokemon))
