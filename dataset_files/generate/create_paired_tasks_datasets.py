"""
Generate the `paired_tasks` datasets: pairs of *similar, confusable* input->output
tasks where the function lives only in the (input, output) relation -- i.e. neither
the input token alone nor the output/label token alone reveals which task it is
(matched input AND output marginals). See DECISIONS.md (mixed-task ICL probe).

Produces, under dataset_files/paired_tasks/:
  - next_number.json / prev_number.json : k -> k+1 / k -> k-1, integers as WORDS
        (digits risk tokeniser clumping; words tokenise per-word). Same input set
        {1..200} for both -> identical input marginal; outputs {2..201}/{0..199}
        -> near-identical output marginal. Only the +1/-1 relation distinguishes.
  - rhyme.json : word -> a phonetically-rhyming word (CMUdict via `pronouncing`).
        Inputs drawn from the synonym/antonym vocabulary and outputs constrained to
        that same real-word vocabulary, so rhyme shares the general-English-word
        input/output marginal with synonym and antonym (-> synonym|rhyme, antonym|rhyme
        are marginal-matched pairs).
  - synonym.json / antonym.json : copied verbatim from dataset_files/abstractive/.

Run:  HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1 \
      python dataset_files/generate/create_paired_tasks_datasets.py
"""
import os
import json
import shutil

from transformers import AutoTokenizer
import pronouncing

ABSTRACTIVE = "dataset_files/abstractive"
OUT = "dataset_files/paired_tasks"
N = 200

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")


def n_tokens_after_colon(w):
    """How many tokens the output `w` adds in the template '...A:{w}'."""
    return len(tok("A:" + w).input_ids) - len(tok("A:").input_ids)


# ---------------------------------------------------------------- number words
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def int_to_words(n):
    """0..999 -> English words, hyphenated tens, no 'and' (e.g. 'one hundred twenty-three')."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + ("-" + _ONES[o] if o else "")
    h, rem = divmod(n, 100)
    return _ONES[h] + " hundred" + (" " + int_to_words(rem) if rem else "")


def build_number_tasks():
    inputs = list(range(1, N + 1))  # same input set for both -> matched input marginal
    nxt = [{"input": int_to_words(k), "output": int_to_words(k + 1)} for k in inputs]
    prv = [{"input": int_to_words(k), "output": int_to_words(k - 1)} for k in inputs]
    return nxt, prv


# ----------------------------------------------------------------------- rhyme
def build_rhyme_task():
    syn = json.load(open(os.path.join(ABSTRACTIVE, "synonym.json")))
    ant = json.load(open(os.path.join(ABSTRACTIVE, "antonym.json")))
    words = ([e["input"] for e in syn] + [e["output"] for e in syn]
             + [e["input"] for e in ant] + [e["output"] for e in ant])
    vocab = set(w.lower() for w in words if w.isalpha())

    def rhyme_in_vocab(w):
        for r in pronouncing.rhymes(w):
            if (r in vocab and r != w and not r.startswith(w) and not w.startswith(r)
                    and n_tokens_after_colon(r) == 1):
                return r
        return None

    pairs, seen = [], set()
    for w in sorted(vocab):  # sorted -> deterministic
        if w in seen or n_tokens_after_colon(w) != 1:
            continue
        r = rhyme_in_vocab(w)
        if r:
            pairs.append({"input": w, "output": r})
            seen.add(w)
        if len(pairs) >= N:
            break
    return pairs


def main():
    os.makedirs(OUT, exist_ok=True)

    nxt_n, prv_n = build_number_tasks()
    rhyme = build_rhyme_task()

    to_write = {
        "next_number.json": nxt_n,
        "prev_number.json": prv_n,
        "rhyme.json": rhyme,
    }
    for fname, data in to_write.items():
        with open(os.path.join(OUT, fname), "w") as f:
            json.dump(data, f, indent=2)
        print(f"wrote {fname:18s} {len(data):4d} pairs   e.g. {data[0]['input']!r} -> {data[0]['output']!r}")

    for fname in ("synonym.json", "antonym.json"):
        shutil.copy(os.path.join(ABSTRACTIVE, fname), os.path.join(OUT, fname))
        n = len(json.load(open(os.path.join(OUT, fname))))
        print(f"copied {fname:18s} {n:4d} pairs")


if __name__ == "__main__":
    main()
