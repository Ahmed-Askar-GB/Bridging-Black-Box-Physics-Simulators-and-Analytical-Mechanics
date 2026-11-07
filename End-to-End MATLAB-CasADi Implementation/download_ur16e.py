import os
import urllib.request
import json

headers = {'User-Agent': 'Mozilla/5.0'}

target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_UR16e")
os.makedirs(target_dir, exist_ok=True)

def download_dir(api_url, local_dst):
    os.makedirs(local_dst, exist_ok=True)
    req = urllib.request.Request(api_url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        items = json.loads(resp.read().decode('utf-8'))
    for it in items:
        if it['type'] == 'file':
            download_url = it['download_url']
            dest_file = os.path.join(local_dst, it['name'])
            print(f"Downloading {it['name']} -> {dest_file}...")
            urllib.request.urlretrieve(download_url, dest_file)
        elif it['type'] == 'dir':
            download_dir(it['url'], os.path.join(local_dst, it['name']))

# 1. Download urdf dir or specifically universalUR16e.urdf
print("Downloading URDF files...")
download_dir("https://api.github.com/repos/Daniella1/urdf_files_dataset/contents/urdf_files/matlab/ur_e_description/urdf", os.path.join(target_dir, "urdf"))

# Also copy universalUR16e.urdf to top-level test_UR16e folder for convenience
urdf_source = os.path.join(target_dir, "urdf", "universalUR16e.urdf")
if os.path.exists(urdf_source):
    import shutil
    shutil.copy2(urdf_source, os.path.join(target_dir, "universalUR16e.urdf"))

# 2. Download meshes dir
print("Downloading Meshes...")
download_dir("https://api.github.com/repos/Daniella1/urdf_files_dataset/contents/urdf_files/matlab/ur_e_description/meshes", os.path.join(target_dir, "meshes"))

# 3. Download meta-information
meta_url = "https://raw.githubusercontent.com/Daniella1/urdf_files_dataset/main/urdf_files/matlab/ur_e_description/meta-information.json"
urllib.request.urlretrieve(meta_url, os.path.join(target_dir, "meta-information.json"))

print("UR16e Download Complete!")
