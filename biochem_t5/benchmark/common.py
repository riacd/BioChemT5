from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase


def _canonicalize_molecule(smiles: str) -> str | None:
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonicalize_smiles_set(smiles: str) -> str | None:
    """Canonicalize a dot-separated molecule set with order-independent parts."""
    text = "".join(str(smiles).split())
    if not text:
        return None
    parts: list[str] = []
    for part in text.split("."):
        if not part:
            return None
        canonical = _canonicalize_molecule(part)
        if canonical is None:
            return None
        parts.append(canonical)
    return ".".join(sorted(parts))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
