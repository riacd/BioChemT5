from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


SMILES_REGEX = re.compile(
    r"(</s>|<pad>|<unk>|<forward>|<retro>|<mlm>|<ec>|<mask>|<mask_product>|<mask_reactants>|<extra_id_\d+>|"
    r"<[A-Za-z][A-Za-z0-9_]*>|"
    r"\[[^\]]+\]|>>|<-|->|"
    r"Bi|Br|Ge|Te|Mo|Mg|Na|Ca|Li|Al|Si|Se|Cl|Fe|Zn|Cu|Mn|Ni|Co|Pd|Pt|"
    r"@@?|%\d{2}|[A-Za-z]|\d|=|#|-|\+|\\|/|\.|:|~|@|\?|\*|\$|\(|\)|<|>)"
)
ATOM_MAP_REGEX = re.compile(r":\d+(?=\])")

BASE_SPECIAL_TOKENS = [
    "<pad>",
    "</s>",
    "<unk>",
    "<forward>",
    "<retro>",
    "<mlm>",
    "<ec>",
    "<mask_product>",
    "<mask_reactants>",
]
SENTINEL_TOKENS = [f"<extra_id_{idx}>" for idx in range(100)]
MASK_TOKEN = "<mask>"


def smiles_tokenize(text: str) -> list[str]:
    compact = "".join(str(text).split())
    if not compact:
        return []
    tokens = SMILES_REGEX.findall(compact)
    if "".join(tokens) != compact:
        # Keep unusual characters visible to the model instead of dropping them.
        return list(compact)
    return tokens


def strip_atom_maps(text: str) -> str:
    return ATOM_MAP_REGEX.sub("", str(text))


def strip_atom_maps_from_tokens(tokens: list[str]) -> list[str]:
    return [ATOM_MAP_REGEX.sub("", token) if token.startswith("[") and token.endswith("]") else token for token in tokens]


class SmilesTokenizer:
    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}
        self.pad_token_id = self.token_to_id["<pad>"]
        self.eos_token_id = self.token_to_id["</s>"]
        self.unk_token_id = self.token_to_id["<unk>"]
        self.mask_token_id = self.token_to_id.get(MASK_TOKEN)

    @classmethod
    def build(cls, texts: Iterable[str], vocab_size: int = 4096) -> "SmilesTokenizer":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(smiles_tokenize(text))
        token_to_id: dict[str, int] = {}
        for token in BASE_SPECIAL_TOKENS + SENTINEL_TOKENS:
            token_to_id.setdefault(token, len(token_to_id))
        for token, _count in counter.most_common():
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)
            if len(token_to_id) >= vocab_size:
                break
        return cls(token_to_id)

    @classmethod
    def load(cls, path: str | Path) -> "SmilesTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({str(k): int(v) for k, v in payload["token_to_id"].items()})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"token_to_id": self.token_to_id}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def __len__(self) -> int:
        return len(self.token_to_id)

    def token_id(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_token_id)

    def add_tokens(self, tokens: Iterable[str]) -> list[str]:
        added: list[str] = []
        for token in tokens:
            if token not in self.token_to_id:
                token_id = len(self.token_to_id)
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
                added.append(token)
        self.mask_token_id = self.token_to_id.get(MASK_TOKEN)
        return added

    def ensure_mask_token(self) -> int:
        self.add_tokens([MASK_TOKEN])
        return self.token_to_id[MASK_TOKEN]

    def tokenize(self, text: str) -> list[str]:
        return smiles_tokenize(text)

    def encode(self, text: str, add_eos: bool = True) -> list[int]:
        ids = [self.token_id(token) for token in self.tokenize(text)]
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = False) -> str:
        tokens: list[str] = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), "<unk>")
            if skip_special and token.startswith("<") and token.endswith(">"):
                continue
            tokens.append(token)
        return "".join(tokens)


def pad_sequences(sequences: list[list[int]], pad_id: int) -> tuple[list[list[int]], list[list[int]]]:
    max_len = max((len(seq) for seq in sequences), default=0)
    padded: list[list[int]] = []
    masks: list[list[int]] = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        padded.append(seq + [pad_id] * pad_len)
        masks.append([1] * len(seq) + [0] * pad_len)
    return padded, masks
