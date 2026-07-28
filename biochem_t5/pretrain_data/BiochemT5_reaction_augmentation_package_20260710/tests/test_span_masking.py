from __future__ import annotations

import random

from biochem_t5.data.span_masking import make_t5_span_corruption, token_weights_from_center_maps
from biochem_t5.data.smiles_tokenizer import smiles_tokenize, strip_atom_maps, strip_atom_maps_from_tokens


def test_center_atom_maps_receive_larger_mask_weights() -> None:
    tokens = ["<mlm>", "[CH3:1]", "[OH:2]", ">>", "[CH2:3]"]
    record = {
        "reaction_center_changed_atom_maps": [1],
        "reaction_center_neighbor_atom_maps": [2],
    }

    weights = token_weights_from_center_maps(tokens, record, base=1.0, center=4.0, neighbor=2.0)

    assert weights == [1.0, 4.0, 2.0, 1.0, 1.0]


def test_t5_span_corruption_uses_sentinels_and_target_eos() -> None:
    tokens = ["<mlm>", "[CH3:1]", "[OH:2]", ">>", "[CH2:3]"]
    weights = [1.0, 4.0, 2.0, 1.0, 1.0]

    corrupted, target, meta = make_t5_span_corruption(tokens, weights, random.Random(3), mask_fraction=0.4)

    assert any(tok.startswith("<extra_id_") for tok in corrupted)
    assert target[-1] == "</s>"
    assert meta["masked_token_count"] >= 1


def test_strip_atom_maps_preserves_token_boundaries() -> None:
    mapped = "[CH3:12][O-:7]>>[CH2:12]=O"
    raw_tokens = smiles_tokenize(mapped)
    stripped_tokens = strip_atom_maps_from_tokens(raw_tokens)

    assert strip_atom_maps(mapped) == "[CH3][O-]>>[CH2]=O"
    assert stripped_tokens == ["[CH3]", "[O-]", ">>", "[CH2]", "=", "O"]
    assert len(raw_tokens) == len(stripped_tokens)


def test_smiles_tokenizer_does_not_treat_dative_fragment_as_special_token() -> None:
    tokens = smiles_tokenize("<-N2=C1->")

    assert tokens == ["<-", "N", "2", "=", "C", "1", "->"]
