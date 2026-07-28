from __future__ import annotations

import random
from functools import lru_cache
from typing import Any

from .smiles_tokenizer import strip_atom_maps

try:
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.warning")
except ImportError:  # pragma: no cover - runtime fallback for minimal environments.
    Chem = None  # type: ignore[assignment]
    rdMolStandardize = None  # type: ignore[assignment]


MASK_PRODUCT = "<mask_product>"
MASK_REACTANTS = "<mask_reactants>"

SMALL_COFACTOR_SMILES = {
    "O",
    "[H+]",
    "[H]",
    "[H][H]",
    "O=O",
    "O=C=O",
    "N",
    "[NH4+]",
    "[OH-]",
    "[O-]",
    "C(=O)(O)O",
    "O=C([O-])O",
    "O=C(O)[O-]",
    "O=P(O)(O)O",
    "O=P([O-])(O)O",
    "O=P([O-])([O-])O",
    "O=S(=O)(O)O",
    "O=S(=O)([O-])[O-]",
    "O=P(O)(O)OP(=O)(O)O",
    "O=P([O-])([O-])OP(=O)([O-])[O-]",
}

LARGE_COFACTOR_SMILES = {
    # Common biochemical cofactors from data/Reaction_gen/molecule_data/kegg_metacyc_merged.tsv.
    "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O",  # ATP
    "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O",  # ADP
    "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O",  # AMP
    "NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1",  # NAD+
    "NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1",  # NADH
    "NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)(O)O)[C@@H]3O)[C@@H](O)[C@H]2O)c1",  # NADP+
    "NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)(O)O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1",  # NADPH
    "CC(C)(COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O)[C@@H](O)C(=O)NCCC(=O)NCCS",  # CoA
    "Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)c2cc1C",  # FAD
    "Cc1cc2c(cc1C)N(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n3cnc4c(N)ncnc43)[C@H](O)[C@@H]1O)c1[nH]c(=O)[nH]c(=O)c1N2",  # FADH2
    "Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)O)c2cc1C",  # FMN
    "Cc1cc2c(cc1C)N(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)O)c1[nH]c(=O)[nH]c(=O)c1N2",  # FMNH2
    "C[S+](CC[C@H](N)C(=O)[O-])C[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O",  # SAM
    "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CSCC[C@H](N)C(=O)O)[C@@H](O)[C@H]1O",  # SAH
    "Nc1nc2c(ncn2[C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)[nH]1",  # GTP
    "Nc1nc2c(ncn2[C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)[nH]1",  # GDP
    "O=c1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)[nH]1",  # UTP
    "O=c1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)[nH]1",  # UDP
    "Nc1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)n1",  # CTP
    "Nc1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)n1",  # CDP
}


def split_reaction(rxn: str) -> tuple[str, str]:
    if ">>" in rxn:
        left, right = rxn.split(">>", 1)
        return left, right
    parts = rxn.split(">")
    if len(parts) == 3:
        return parts[0], parts[2]
    raise ValueError(f"Invalid reaction: {rxn[:120]}")


def split_molecules(side: str) -> list[str]:
    return [part for part in str(side).split(".") if part]


def heavy_atom_score(smiles: str) -> int:
    return sum(1 for char in smiles if char.isalpha() and char.isupper() and char != "H")


@lru_cache(maxsize=8192)
def _canonical_forms(smiles: str) -> tuple[str, ...]:
    if Chem is None:
        return ()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ()
    forms = {Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)}
    if rdMolStandardize is not None:
        try:
            uncharged = rdMolStandardize.Uncharger().uncharge(mol)
            forms.add(Chem.MolToSmiles(uncharged, canonical=True, isomericSmiles=True))
        except (RuntimeError, ValueError):
            pass
    return tuple(forms)


@lru_cache(maxsize=1)
def _canonical_cofactor_forms() -> set[str]:
    forms: set[str] = set()
    for smiles in SMALL_COFACTOR_SMILES | LARGE_COFACTOR_SMILES:
        forms.update(_canonical_forms(smiles))
    return forms


def is_common_cofactor(smiles: str) -> bool:
    if smiles in SMALL_COFACTOR_SMILES:
        return True
    if _canonical_forms(smiles) and set(_canonical_forms(smiles)).intersection(_canonical_cofactor_forms()):
        return True
    return heavy_atom_score(smiles) <= 1


def cofactor_aware_product_key(smiles: str) -> tuple[int, int]:
    return (0 if is_common_cofactor(smiles) else 1, heavy_atom_score(smiles))


def choose_non_cofactor_product(product_side: str) -> str:
    products = split_molecules(product_side)
    if not products:
        return product_side
    return max(products, key=cofactor_aware_product_key)


def canonical_view_from_record(record: dict[str, Any]) -> dict[str, str]:
    reactants, products = split_reaction(str(record["rxn"]))
    return {
        "forward_input": strip_atom_maps(reactants),
        "forward_target": strip_atom_maps(products),
        "retro_input": strip_atom_maps(products),
        "retro_target": strip_atom_maps(reactants),
    }


def available_views(record: dict[str, Any]) -> list[dict[str, str]]:
    views = record.get("rsmiles_views") or []
    out: list[dict[str, str]] = []
    for view in views:
        if not isinstance(view, dict):
            continue
        required = ("forward_input", "forward_target", "retro_input", "retro_target")
        if all(isinstance(view.get(key), str) and view.get(key) for key in required):
            out.append({key: strip_atom_maps(str(view[key])) for key in required})
    if out:
        return out
    return [canonical_view_from_record(record)]


def sample_view(record: dict[str, Any], rng: random.Random) -> dict[str, str]:
    views = available_views(record)
    return views[rng.randrange(len(views))]


def full_reaction_from_view(view: dict[str, str]) -> str:
    return f"{view['forward_input']}>>{view['forward_target']}"


def build_forward_sample(record: dict[str, Any], rng: random.Random) -> tuple[str, str]:
    view = sample_view(record, rng)
    source = f"<forward>{view['forward_input']}>>{MASK_PRODUCT}"
    target = view["forward_target"]
    return source, target


def build_retro_sample(record: dict[str, Any], rng: random.Random) -> tuple[str, str]:
    view = sample_view(record, rng)
    product = choose_non_cofactor_product(view["retro_input"])
    source = f"<retro>{product}"
    target = view["retro_target"]
    return source, target


def build_mlm_reaction(record: dict[str, Any], rng: random.Random, use_mapped_rxn: bool = True) -> str:
    mapped = record.get("mapped_rxn")
    if use_mapped_rxn and isinstance(mapped, str) and mapped:
        return f"<mlm>{mapped}"
    return f"<mlm>{full_reaction_from_view(sample_view(record, rng))}"


def build_ec_reaction_views(record: dict[str, Any], rng: random.Random) -> tuple[str, str]:
    views = available_views(record)
    if len(views) >= 2:
        first, second = rng.sample(views, 2)
        return f"<ec>{full_reaction_from_view(first)}", f"<ec>{full_reaction_from_view(second)}"
    canonical = canonical_view_from_record(record)
    only = views[0]
    return f"<ec>{full_reaction_from_view(only)}", f"<ec>{full_reaction_from_view(canonical)}"


def build_ec_reaction_view(record: dict[str, Any], rng: random.Random) -> str:
    return f"<ec>{full_reaction_from_view(sample_view(record, rng))}"


def iter_vocab_texts(record: dict[str, Any]) -> list[str]:
    texts = [strip_atom_maps(str(record.get("rxn", "")))]
    mapped = record.get("mapped_rxn")
    if isinstance(mapped, str):
        texts.append(strip_atom_maps(mapped))
    for view in available_views(record):
        texts.extend(
            [
                f"<forward>{view['forward_input']}>>{MASK_PRODUCT}",
                view["forward_target"],
                f"<retro>{choose_non_cofactor_product(view['retro_input'])}",
                view["retro_target"],
                f"<ec>{full_reaction_from_view(view)}",
            ]
        )
    return texts
