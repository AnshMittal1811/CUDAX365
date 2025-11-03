import json
import os

locations = []
if os.path.exists("../306_news_parse/locations.txt"):
    locations = [line.strip() for line in open("../306_news_parse/locations.txt", "r", encoding="utf-8")]

if os.path.exists("map_db.json"):
    db = json.load(open("map_db.json", "r", encoding="utf-8"))
else:
    db = {"buildings": []}

for loc in locations:
    db["buildings"].append({"location": loc, "source": "news"})

with open("map_db.json", "w", encoding="utf-8") as f:
    json.dump(db, f, indent=2)

print("Wrote map_db.json")
