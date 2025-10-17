import requests
pdb_code = "1a0f"
url = f"https://files.rcsb.org/download/{pdb_code.upper()}.pdb"
r = requests.get(url)
print(r.status_code)
print(r.text[:200])