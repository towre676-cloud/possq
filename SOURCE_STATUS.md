# Source status

Exact claims concern the specified generators, uncertainty intervals, and acquisition protocols. Independent copies and one continuous trajectory must remain separate.

This is a source-preservation import. Run `python restore_evidence.py` before historical replay: large compressible data are stored losslessly in `_evidence/`, keyed by SHA-256. The original path and hash of each file are recorded in `IMPORT_MANIFEST.json`. Externally retained proof streams must be recovered from their original archive before a replay that needs them. Original manifests remain unchanged.

Fresh checks are recorded in `FRESH_VALIDATION.json`; archived receipts are not fresh verification.
