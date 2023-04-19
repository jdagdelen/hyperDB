import json
from hyperdb import HyperDB

# Load documents from the JSONL file
documents = []

with open("pokemon.jsonl", "r") as f:
    for line in f:
        documents.append(json.loads(line))

# Instantiate HyperDB with the list of documents and the key "description"
db = HyperDB(documents, key="info.description")

# Save the HyperDB instance to a file
db.save("pokemon_hyperdb.pickle.gz")

# Load the HyperDB instance from the file
db.load("pokemon_hyperdb.pickle.gz")

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
for result in results:
    print(format_entry(result))
