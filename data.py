from pathlib import Path
import json

# Path where the JSON inventory files are stored
inventory_dir = Path("data/inventories")  # Update this if you used a different path

# File names of all 4 inventories
files = [
    "1900121_meta_inventory.json",
    "1900121_prof_inventory.json",
    "1900121_Rtraj_inventory.json",
    "1900121_tech_inventory.json"
]

# Load each JSON file into a Python dictionary
inventories = {}
for file in files:
    with open(inventory_dir / file, "r", encoding="utf-8") as f:
        inventories[file] = json.load(f)

# Confirm loaded files
print("Loaded inventories:", list(inventories.keys()))
