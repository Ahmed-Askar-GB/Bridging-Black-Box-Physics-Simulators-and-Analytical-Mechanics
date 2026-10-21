import os
import urllib.request
import json

base_api = "https://api.github.com/repos/Daniella1/urdf_files_dataset/contents/urdf_files/matlab/ur_e_description"
req = urllib.request.Request(base_api, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as resp:
    items = json.loads(resp.read().decode('utf-8'))

print("Items in ur_e_description:")
for it in items:
    print(f"  {it['type']}: {it['name']} -> {it['path']}")
