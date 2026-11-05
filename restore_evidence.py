import pathlib,json,gzip,hashlib
r=pathlib.Path(__file__).resolve().parent
for row in json.loads((r/"IMPORT_MANIFEST.json").read_text())["files"]:
 if row.get("storage")=="gzip":
  b=gzip.decompress((r/row["stored_path"]).read_bytes())
  assert hashlib.sha256(b).hexdigest()==row["sha256"], row["path"]
  p=r/row["path"];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
print("Restored and hash-verified all Git-hosted compressed evidence. See manifest for externally retained proof streams.")
