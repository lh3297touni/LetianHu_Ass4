# app/bigram_model.py
from __future__ import annotations
from collections import Counter, defaultdict
from typing import Iterable, List, Dict
import random
import re


class BigramModel:

    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(self, corpus: Iterable[str], lowercase: bool = True, seed: int | None = 42):
        self.lowercase = lowercase
        self.random = random.Random(seed)

        # token
        tokenized = [self._tokenize(s) for s in corpus]

        # vocab
        vocab = Counter(tok for sent in tokenized for tok in sent)
        self.vocab = set(vocab.keys()) | {self.BOS, self.EOS, self.UNK}

        # count
        self.unigram: Counter = Counter()
        self.bigram: Dict[str, Counter] = defaultdict(Counter)
        for sent in tokenized:
            seq = [self.BOS] + sent + [self.EOS]
            for i in range(len(seq) - 1):
                w1, w2 = seq[i], seq[i + 1]
                self.bigram[w1][w2] += 1
                self.unigram[w1] += 1
            self.unigram[seq[-1]] += 1

        self.V = len(self.vocab)

    # public API 
    def generate(self, start_word: str, length: int = 10) -> str:
  
        prev = self._norm(start_word)
        if prev not in self.vocab:
            prev = self.UNK

        out: List[str] = []
        if prev not in (self.BOS, self.EOS):
            out.append(prev)

        # sample
        while len(out) < max(0, length):
            nxt = self._sample_next(prev)
            if nxt == self.EOS:
                break
            if nxt not in (self.BOS, self.EOS):
                out.append(nxt)
            prev = nxt

        return self._detokenize(out)

    # internals
    def _tokenize(self, s: str) -> List[str]:
        if self.lowercase:
            s = s.lower()
        return re.findall(r"[a-zA-Z0-9']+|[.,!?;:]", s)

    def _detokenize(self, toks: List[str]) -> str:
        pieces: List[str] = []
        for i, t in enumerate(toks):
            if i > 0 and t not in ".,!?;:":
                pieces.append(" ")
            pieces.append(t)
        return "".join(pieces)

    def _norm(self, w: str) -> str:
        return w.lower() if self.lowercase else w

    def _probs(self, w1: str) -> Dict[str, float]:
        if w1 not in self.bigram:
            w1 = self.UNK
        counts = self.bigram[w1]
        denom = self.unigram.get(w1, 0) + self.V
        return {w2: (counts[w2] + 1) / denom for w2 in self.vocab}

    def _sample_next(self, w1: str) -> str:
        probs = self._probs(w1)
        r = self.random.random()
        cum = 0.0
        for w2, p in probs.items():
            cum += p
            if r <= cum:
                return w2
        return self.EOS
