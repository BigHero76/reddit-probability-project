import json

BASE = "/home/claude/project/reddit-probability-project"

with open(f"{BASE}/data-pipeline/output/map.json") as f:
    map_data = json.load(f)
with open(f"{BASE}/data-pipeline/output/transitions.json") as f:
    trans_data = json.load(f)

full_nodes = map_data["nodes"]
hopper_nodes = [
    {"subreddit": n["subreddit"], "x": n["x"], "y": n["y"], "stationary_prob": n["stationary_prob"]}
    for n in full_nodes
]
edges = trans_data["edges"]

# ---- These get filled with real published URLs after each page is
# published once; placeholders keep the templates reusable in the
# meantime (relative filenames work fine for the local/zip version). ----
NAV_URLS = {
    "INDEX_URL": "index.html",
    "MAP_URL": "map.html",
    "HOPPER_URL": "hopper.html",
}

def inject_nav(html):
    for key, val in NAV_URLS.items():
        html = html.replace(key, val)
    return html

# ---- Map page ----
with open(f"{BASE}/frontend/map_template.html") as f:
    map_html = f.read()
map_html = map_html.replace("__MAP_DATA__", json.dumps(map_data))
map_html = inject_nav(map_html)
with open(f"{BASE}/frontend/map.html", "w") as f:
    f.write(map_html)
print(f"map.html written, {len(map_html)/1024:.0f} KB")

# ---- Hopper page ----
with open(f"{BASE}/frontend/hopper_template.html") as f:
    hopper_html = f.read()
hopper_html = hopper_html.replace("__MAP_NODES__", json.dumps(hopper_nodes))
hopper_html = hopper_html.replace("__EDGES__", json.dumps(edges))
hopper_html = inject_nav(hopper_html)
with open(f"{BASE}/frontend/hopper.html", "w") as f:
    f.write(hopper_html)
print(f"hopper.html written, {len(hopper_html)/1024:.0f} KB")
