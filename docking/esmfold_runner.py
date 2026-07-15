#!/usr/bin/env python3
"""
ESMFold Runner 
"""
import sys
import os
import math
import types

import torch
import openfold
_stub = types.ModuleType("attn_core_inplace_cuda")
_stub.forward_ = lambda *a, **k: None
_stub.backward_ = lambda *a, **k: None
sys.modules["attn_core_inplace_cuda"] = _stub

import esm


_model = None

print(f"[DEBUG] openfold loaded from: {openfold.__file__}", file=sys.stderr)
def load_model():
    global _model
    if _model is not None:
        return _model
    

    
    print("Loading ESMFold on GPU...")
    model = esm.pretrained.esmfold_v1()
    #model = model.eval().cuda()
    model = model.eval()
    model.esm.half()  # fp16 for the ESM-2 LM backbone; trunk stays fp32 for stability
    model = model.cuda()
    model.trunk.set_chunk_size(64)  # lowe less memory, slower
    _model = model
    print("ESMFold model loaded successfully (fp16 LM + chunked trunk)")
    return model


def fold_sequence(seq, output_file):
    """Fold peptide sequence and save valid PDB. Returns absolute path."""
    model = load_model()
    print(f"Running ESMFold inference ({len(seq)} residues)...")

    #with torch.no_grad():
        #output = model.infer(seq)
        #pdb_string = model.output_to_pdb(output)[0]
        
    with torch.no_grad():
        output = model.infer(seq)
        pdb_string = model.output_to_pdb(output)[0]
    del output
    torch.cuda.empty_cache()

    if not pdb_string or "ATOM" not in pdb_string:
        raise RuntimeError(
            f"ESMFold returned empty/invalid PDB for sequence: {seq[:30]}...\n"
            f"Output: {str(pdb_string)[:300]}"
        )

    bad_coords = 0
    for line in pdb_string.splitlines():
        if line.startswith("ATOM") and len(line) >= 54:
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                if any(math.isnan(v) or math.isinf(v) for v in [x, y, z]):
                    bad_coords += 1
            except ValueError:
                bad_coords += 1

    if bad_coords > 0:
        raise RuntimeError(
            f"ESMFold produced {bad_coords} atoms with NaN/Inf coordinates."
        )

    atom_count = sum(1 for l in pdb_string.splitlines() if l.startswith("ATOM"))
    print(f"ESMFold generated {atom_count} valid ATOM records")

    out_abs = os.path.abspath(output_file)
    out_dir = os.path.dirname(out_abs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_abs, "w") as f:
        f.write(pdb_string)
        stripped = pdb_string.rstrip()
        if not stripped.endswith("END"):
            if not stripped.endswith("TER"):
                f.write("\nTER\n")
            f.write("END\n")

    print(f"PDB saved → {out_abs}")
    return out_abs


if __name__ == "__main__":
    seq         = sys.argv[1]
    output_file = sys.argv[2]

    print(f"Sequence ({len(seq)} aa): {seq}")
    print(f"Output file: {output_file}")

    result = fold_sequence(seq, output_file)
    print(f"Done → {result}")
