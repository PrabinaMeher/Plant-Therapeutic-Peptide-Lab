#!/usr/bin/env python3
"""
fold.py - minimal CLI shim inside the ptplab-esmfold container.

Usage: python fold.py <SEQUENCE> <OUTPUT_PDB_PATH>

Called by compute_backend.run_esmfold() via `docker exec ptplab-esmfold
python /app/fold.py <seq> /tmp/out.pdb`, then the file is copied out with
`docker cp`.

NOTE: This loads the ESM model fresh on every call for simplicity. For
production use with frequent folding requests, wrap this in a small FastAPI
server inside the container instead (load model once, serve over HTTP) -
swap compute_backend.run_esmfold's `docker exec` call for an HTTP POST to
that server. Left as a CLI here to keep the container's surface area small
and easy to reason about / swap with your existing pipeline.fold_peptide
logic.
"""

import sys

import torch
import esm


def fold(sequence: str, out_path: str):
    model = esm.pretrained.esmfold_v1()
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    with torch.no_grad():
        output = model.infer_pdb(sequence)

    with open(out_path, "w") as f:
        f.write(output)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fold.py <SEQUENCE> <OUTPUT_PDB_PATH>", file=sys.stderr)
        sys.exit(1)

    seq, out_path = sys.argv[1], sys.argv[2]
    fold(seq, out_path)
    print(f"Wrote {out_path}")
