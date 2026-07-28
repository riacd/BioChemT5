from __future__ import annotations

import json

from biochem_t5.data.finalize_unique_rsmiles import dedupe_record_views


def test_dedupe_record_views_removes_duplicates_without_padding() -> None:
    record = {
        "rxn": "CCO>>CC=O",
        "rsmiles_status": "ok",
        "rsmiles_views": [
            {"forward_input": "CCO", "forward_target": "CC=O", "retro_input": "CC=O", "retro_target": "CCO"},
            {"forward_input": "CCO", "forward_target": "CC=O", "retro_input": "CC=O", "retro_target": "CCO"},
            {"forward_input": "OCC", "forward_target": "O=CC", "retro_input": "O=CC", "retro_target": "OCC"},
        ],
    }

    finalized, removed = dedupe_record_views(record, max_views=20)

    assert finalized["rsmiles_view_count"] == 2
    assert finalized["unique_rsmiles_view_count"] == 2
    assert len(finalized["rsmiles_views"]) == 2
    assert removed == 1


def test_dedupe_record_views_can_be_json_serialized() -> None:
    finalized, _removed = dedupe_record_views({"rxn": "CCO>>CC=O", "rsmiles_status": "ok", "rsmiles_views": []}, max_views=20)

    json.dumps(finalized, ensure_ascii=False)
