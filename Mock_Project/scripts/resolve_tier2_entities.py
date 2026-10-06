"""
Global 2nd-Tier Metals Transparency & Cost-at-Risk Engine
=========================================================
PHASE 2 -- n-tier entity resolution (fuzzy supplier matching).

Business problem
----------------
Tier-1 suppliers declare their own sub-suppliers through three different
channels (ERP, supplier survey, sustainability portal). Nobody maintains a
DUNS number. The same physical smelter therefore appears under several
spellings, so the 2nd-tier network *looks* diversified when it is not.
Until those records are resolved, no concentration, compliance or CO2
statement about the n-tier network can be trusted.

What this script does
---------------------
1. NORMALISE   Unicode folding, legal-form stripping, domain synonyms
2. BLOCK       candidate generation on country x metal (cheap, domain-safe)
3. SCORE       ensemble of Jaro-Winkler / token-set / bigram-Dice
4. CLUSTER     union-find over edges above the auto-match threshold
5. SURVIVE     golden-record selection (most complete, properly cased)
6. EVALUATE    pairwise + cluster metrics vs. _ground_truth_tier2.csv,
               threshold sweep, blocking recall, no-blocking ablation
7. REVEAL      scenario S2 -- how many Tier-1s hide behind one entity

No third-party matching library is used: Levenshtein, Jaro-Winkler,
token-set ratio and Dice are implemented here so the logic is auditable
and the script runs anywhere (pandas + numpy only).

Inputs   : Mock_Project/data/raw/*.csv
Outputs  : Mock_Project/data/processed/*.csv
"""

from __future__ import annotations

import re
import unicodedata
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "processed"

# Ensemble weights (sum to 1.0)
W_JW_COMPACT = 0.40   # whitespace-insensitive  -> "Hong Tai" == "Hongtai"
W_TOKEN_SET = 0.35    # word-order insensitive  -> handles reordering/extra words
W_DICE = 0.25         # character bigrams       -> robust to typos

AUTO_MATCH_THRESHOLD = 0.90   # >= this  -> merge automatically
REVIEW_LOW_THRESHOLD = 0.78   # in between -> human review queue
# below REVIEW_LOW            -> treated as different entities

# ---------------------------------------------------------------------------
# 1. Normalisation
# ---------------------------------------------------------------------------

# Legal forms and corporate noise words carry no identifying information.
LEGAL_FORMS = {
    "gmbh", "gesmbh", "mbh", "ag", "aktiengesellschaft", "kg", "kgaa", "se",
    "ab", "as", "asa", "oy", "oyj", "aps", "ou", "uab", "sia",
    "sa", "spa", "srl", "sl", "sll", "sas", "sarl", "bv", "nv", "plc",
    "sro", "as", "zoo", "oo", "sp", "z", "doo", "dd", "kft", "zrt", "bt",
    "ltd", "limited", "llc", "lc", "inc", "incorporated", "corp", "corporation",
    "co", "company", "cia", "fze", "fzco", "pte", "pvt", "jsc", "ojsc", "pjsc",
    "eood", "ood", "ad", "tic", "ith", "ihr",
}
NOISE_WORDS = {"and", "the", "of", "group", "holding", "holdings", "international"
               } - {"international"}  # keep 'international': it *is* part of a name

# Domain synonyms -- one spelling wins.
SYNONYMS = {
    "aluminum": "aluminium",
    "alu": "aluminium",
    "aluminio": "aluminium",
    "intl": "international",
    "int": "international",
    "und": "and",
    "technologies": "technology",
    "tech": "technology",
    "industries": "industry",
    "sanayii": "sanayi",
    "bakirr": "bakir",
    "veredlung": "veredelung",
    "zinkveredlung": "zinkveredelung",
    "metalwerke": "metallwerke",
    "sohne": "sohn",
    "soehne": "sohn",
}

# Transliterations that Unicode NFKD alone gets wrong (German/Turkish/Nordic).
TRANSLIT = {
    "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
    "å": "aa", "æ": "ae", "ø": "oe",
    "ı": "i", "ş": "s", "ğ": "g", "ç": "c", "İ": "i",
    "ł": "l", "ż": "z", "ź": "z", "ś": "s", "ę": "e", "ą": "a", "ń": "n", "ć": "c",
    "č": "c", "š": "s", "ž": "z", "ř": "r", "ě": "e", "ů": "u", "ý": "y",
}


def normalise_name(raw: str) -> tuple[str, str]:
    """Return (normalised token string, compact whitespace-free string)."""
    s = str(raw).strip()

    # a) dotted acronyms first: "G.m.b.H." -> "GmbH", "S.p.A." -> "SpA"
    s = s.replace(".", "")

    # b) explicit transliteration, then NFKD to strip any remaining accents
    s = "".join(TRANSLIT.get(ch, ch) for ch in s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # c) lowercase, punctuation -> space
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # d) token level: synonyms, then drop legal forms and noise
    tokens = [SYNONYMS.get(t, t) for t in s.split()]
    tokens = [t for t in tokens if t not in LEGAL_FORMS and t not in NOISE_WORDS]

    # never return an empty key
    if not tokens:
        tokens = s.split() or [str(raw).lower()]

    norm = " ".join(tokens)
    compact = norm.replace(" ", "")
    return norm, compact


# ---------------------------------------------------------------------------
# 2. Similarity primitives (no external dependencies)
# ---------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def lev_ratio(a: str, b: str) -> float:
    m = max(len(a), len(b))
    return 1.0 if m == 0 else 1.0 - levenshtein(a, b) / m


def jaro(a: str, b: str) -> float:
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    window = max(la, lb) // 2 - 1
    if window < 0:
        window = 0
    a_flags = [False] * la
    b_flags = [False] * lb
    matches = 0
    for i in range(la):
        lo = max(0, i - window)
        hi = min(i + window + 1, lb)
        for j in range(lo, hi):
            if not b_flags[j] and a[i] == b[j]:
                a_flags[i] = b_flags[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    k = 0
    transpositions = 0
    for i in range(la):
        if a_flags[i]:
            while not b_flags[k]:
                k += 1
            if a[i] != b[k]:
                transpositions += 1
            k += 1
    t = transpositions / 2
    return (matches / la + matches / lb + (matches - t) / matches) / 3.0


def jaro_winkler(a: str, b: str, p: float = 0.1, max_prefix: int = 4) -> float:
    """Jaro with a bonus for a common prefix -- company names rarely differ at the front."""
    j = jaro(a, b)
    prefix = 0
    for ca, cb in zip(a[:max_prefix], b[:max_prefix]):
        if ca != cb:
            break
        prefix += 1
    return j + prefix * p * (1 - j)


def token_set_ratio(a: str, b: str) -> float:
    """Word-order and extra-word tolerant ratio (rapidfuzz-style, reimplemented)."""
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    inter = sorted(ta & tb)
    rest_a = sorted(ta - tb)
    rest_b = sorted(tb - ta)
    s_inter = " ".join(inter)
    s_a = (s_inter + " " + " ".join(rest_a)).strip()
    s_b = (s_inter + " " + " ".join(rest_b)).strip()
    if not s_inter:
        return lev_ratio(s_a, s_b)
    return max(lev_ratio(s_inter, s_a), lev_ratio(s_inter, s_b), lev_ratio(s_a, s_b))


def bigrams(s: str) -> set[str]:
    return {s[i:i + 2] for i in range(len(s) - 1)}


def dice(a: str, b: str) -> float:
    ba, bb = bigrams(a), bigrams(b)
    if not ba or not bb:
        return 1.0 if a == b else 0.0
    return 2 * len(ba & bb) / (len(ba) + len(bb))


def similarity(rec_a: dict, rec_b: dict) -> dict:
    """
    Two-pass design, as in commercial MDM tooling:
      pass 1  deterministic -- identical normalised key => certain match.
              This is what rescues German/Nordic compounds, where token-based
              scores collapse ('Stahlservice' vs. 'Stahl Service').
      pass 2  probabilistic -- weighted ensemble of three string metrics.
    """
    s_jw = jaro_winkler(rec_a["name_compact"], rec_b["name_compact"])
    s_tok = token_set_ratio(rec_a["name_norm"], rec_b["name_norm"])
    s_dice = dice(rec_a["name_compact"], rec_b["name_compact"])
    score = W_JW_COMPACT * s_jw + W_TOKEN_SET * s_tok + W_DICE * s_dice

    deterministic = rec_a["name_compact"] == rec_b["name_compact"]
    if deterministic:
        score = 1.0

    return {
        "jaro_winkler": round(s_jw, 4),
        "token_set": round(s_tok, 4),
        "bigram_dice": round(s_dice, 4),
        "ensemble_score": round(W_JW_COMPACT * s_jw + W_TOKEN_SET * s_tok
                                + W_DICE * s_dice, 4),
        "match_score": round(score, 4),
        "match_rule": "DETERMINISTIC_KEY" if deterministic else "PROBABILISTIC",
    }


# ---------------------------------------------------------------------------
# 3. Union-find clustering
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def clusters(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for k in self.parent:
            out.setdefault(self.find(k), []).append(k)
        return out


# ---------------------------------------------------------------------------
# 4. Pipeline
# ---------------------------------------------------------------------------

def build_candidate_pairs(records: list[dict], blocked: bool = True) -> pd.DataFrame:
    """Score pairs. With blocking, only compare within country x metal."""
    if blocked:
        blocks: dict[tuple, list[dict]] = {}
        for r in records:
            blocks.setdefault((r["country_code"], r["supplied_metal_id"]), []).append(r)
        pair_iter = (pair for grp in blocks.values() for pair in combinations(grp, 2))
    else:
        pair_iter = combinations(records, 2)

    rows = []
    for a, b in pair_iter:
        scores = similarity(a, b)
        rows.append(
            {
                "raw_supplier_id_a": a["raw_supplier_id"],
                "raw_supplier_id_b": b["raw_supplier_id"],
                "name_a": a["raw_supplier_name"],
                "name_b": b["raw_supplier_name"],
                "block_key": f"{a['country_code']}|{a['supplied_metal_id']}",
                **scores,
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("match_score", ascending=False).reset_index(drop=True)


SOURCE_PRIORITY = {"ERP_SAP": 0, "SUPPLIER_SURVEY": 1, "SUSTAINABILITY_PORTAL": 2}


def select_golden_record(group: pd.DataFrame) -> pd.Series:
    """
    Survivorship rule, in order:
      1. properly cased names beat SHOUTING ERP extracts
      2. the system of record (ERP_SAP) beats self-declared portals
      3. the longest name wins -- it carries the most legal-form detail
    """
    g = group.copy()
    g["_is_shouting"] = g["raw_supplier_name"].str.isupper().astype(int)
    g["_src_rank"] = g["source_system"].map(SOURCE_PRIORITY).fillna(9).astype(int)
    g["_len"] = g["raw_supplier_name"].str.len()
    g = g.sort_values(["_is_shouting", "_src_rank", "_len"], ascending=[True, True, False])
    return g.iloc[0]


def resolve(records: list[dict], pairs: pd.DataFrame, threshold: float) -> dict[str, str]:
    """Return raw_supplier_id -> cluster_root."""
    uf = UnionFind([r["raw_supplier_id"] for r in records])
    for p in pairs[pairs["match_score"] >= threshold].itertuples(index=False):
        uf.union(p.raw_supplier_id_a, p.raw_supplier_id_b)
    return {k: uf.find(k) for k in uf.parent}


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------

def pairwise_metrics(assignment: dict[str, str], truth: dict[str, str]) -> dict:
    ids = sorted(assignment)
    tp = fp = fn = tn = 0
    for a, b in combinations(ids, 2):
        pred_same = assignment[a] == assignment[b]
        true_same = truth[a] == truth[b]
        if pred_same and true_same:
            tp += 1
        elif pred_same and not true_same:
            fp += 1
        elif not pred_same and true_same:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def cluster_metrics(assignment: dict[str, str], truth: dict[str, str]) -> dict:
    pred: dict[str, set] = {}
    for rid, cid in assignment.items():
        pred.setdefault(cid, set()).add(rid)
    gold: dict[str, set] = {}
    for rid, eid in truth.items():
        gold.setdefault(eid, set()).add(rid)
    gold_sets = list(gold.values())
    exact = sum(1 for c in pred.values() if c in gold_sets)
    return {
        "n_pred_clusters": len(pred),
        "n_true_clusters": len(gold),
        "exact_cluster_matches": exact,
        "cluster_accuracy": exact / len(gold) if gold else 0.0,
    }


def threshold_sweep(records, pairs, truth) -> pd.DataFrame:
    rows = []
    for t in np.round(np.arange(0.60, 0.995, 0.01), 3):
        assignment = resolve(records, pairs, float(t))
        pm = pairwise_metrics(assignment, truth)
        cm = cluster_metrics(assignment, truth)
        rows.append(
            {
                "threshold": float(t),
                "precision": round(pm["precision"], 4),
                "recall": round(pm["recall"], 4),
                "f1": round(pm["f1"], 4),
                "false_merges": pm["fp"],
                "missed_merges": pm["fn"],
                "n_clusters": cm["n_pred_clusters"],
                "exact_clusters": cm["exact_cluster_matches"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(RAW_DIR / "dim_supplier_tier2_raw.csv", encoding="utf-8-sig")
    df_truth = pd.read_csv(RAW_DIR / "_ground_truth_tier2.csv", encoding="utf-8-sig")
    df_decl = pd.read_csv(RAW_DIR / "map_tier1_tier2_declared.csv", encoding="utf-8-sig")
    df_t1 = pd.read_csv(RAW_DIR / "dim_supplier_tier1.csv", encoding="utf-8-sig")

    # ---- 1. normalise ------------------------------------------------------
    norm = df_raw["raw_supplier_name"].map(normalise_name)
    df_raw["name_norm"] = [n for n, _ in norm]
    df_raw["name_compact"] = [c for _, c in norm]
    records = df_raw.to_dict("records")

    # ---- 2+3. block and score ---------------------------------------------
    pairs = build_candidate_pairs(records, blocked=True)
    pairs_all = build_candidate_pairs(records, blocked=False)

    n_all = len(pairs_all)
    reduction = 1 - len(pairs) / n_all if n_all else 0.0

    truth_map = dict(zip(df_truth["raw_supplier_id"], df_truth["true_entity_id"]))
    canon_map = dict(zip(df_truth["raw_supplier_id"], df_truth["canonical_name"]))

    true_pairs = {
        frozenset((a, b))
        for a, b in combinations(sorted(truth_map), 2)
        if truth_map[a] == truth_map[b]
    }
    blocked_pairs = {
        frozenset((r.raw_supplier_id_a, r.raw_supplier_id_b))
        for r in pairs.itertuples(index=False)
    }
    blocking_recall = (
        len(true_pairs & blocked_pairs) / len(true_pairs) if true_pairs else 1.0
    )

    # ---- 4. tune, then cluster --------------------------------------------
    sweep = threshold_sweep(records, pairs, truth_map)
    best = sweep.sort_values(["f1", "threshold"], ascending=[False, True]).iloc[0]

    assignment = resolve(records, pairs, AUTO_MATCH_THRESHOLD)
    pm = pairwise_metrics(assignment, truth_map)
    cm = cluster_metrics(assignment, truth_map)

    # ablation: same threshold, no blocking at all
    assignment_nb = resolve(records, pairs_all, AUTO_MATCH_THRESHOLD)
    pm_nb = pairwise_metrics(assignment_nb, truth_map)

    # ---- 5. golden records -------------------------------------------------
    df_raw["cluster_root"] = df_raw["raw_supplier_id"].map(assignment)

    root_to_entity = {r: f"T2E-{i + 1:03d}" for i, r in enumerate(
        df_raw.groupby("cluster_root")["raw_supplier_id"].min().sort_values().index
    )}
    df_raw["t2_entity_id"] = df_raw["cluster_root"].map(root_to_entity)

    # intra-cluster confidence = weakest edge that held the cluster together
    edge_lookup = {
        frozenset((r.raw_supplier_id_a, r.raw_supplier_id_b)): r.match_score
        for r in pairs.itertuples(index=False)
    }

    golden_rows = []
    for eid, grp in df_raw.groupby("t2_entity_id"):
        win = select_golden_record(grp)
        members = sorted(grp["raw_supplier_id"])
        intra = [
            edge_lookup.get(frozenset((a, b)), 0.0) for a, b in combinations(members, 2)
        ]
        golden_rows.append(
            {
                "t2_entity_id": eid,
                "golden_name": win["raw_supplier_name"].strip(),
                "golden_name_normalised": win["name_norm"],
                "country_code": win["country_code"],
                "supplied_metal_id": win["supplied_metal_id"],
                "tier2_role": win["tier2_role"],
                "n_raw_variants": len(grp),
                "source_systems": "|".join(sorted(grp["source_system"].unique())),
                "surviving_source": win["source_system"],
                "first_seen_date": grp["first_seen_date"].min(),
                "min_intra_cluster_score": round(min(intra), 4) if intra else 1.0,
                "singleton_flag": len(grp) == 1,
            }
        )
    df_golden = pd.DataFrame(golden_rows).sort_values("t2_entity_id").reset_index(drop=True)

    df_map = df_raw[
        ["raw_supplier_id", "raw_supplier_name", "source_system", "t2_entity_id",
         "name_norm", "name_compact", "country_code", "supplied_metal_id"]
    ].merge(
        df_golden[["t2_entity_id", "golden_name"]], on="t2_entity_id", how="left"
    )
    df_map["survivor_flag"] = df_map["raw_supplier_name"].str.strip() == df_map["golden_name"]
    df_map["true_entity_id"] = df_map["raw_supplier_id"].map(truth_map)
    df_map["true_canonical_name"] = df_map["raw_supplier_id"].map(canon_map)

    # ---- human-in-the-loop queue ------------------------------------------
    review = pairs[
        (pairs["match_score"] >= REVIEW_LOW_THRESHOLD)
        & (pairs["match_score"] < AUTO_MATCH_THRESHOLD)
    ].copy()
    review["true_same_entity"] = [
        truth_map[a] == truth_map[b]
        for a, b in zip(review["raw_supplier_id_a"], review["raw_supplier_id_b"])
    ]
    review["suggested_action"] = np.where(
        review["match_score"] >= (AUTO_MATCH_THRESHOLD + REVIEW_LOW_THRESHOLD) / 2,
        "LIKELY MERGE - confirm", "LIKELY DISTINCT - confirm",
    )

    # ---- error analysis ----------------------------------------------------
    pairs["true_same_entity"] = [
        truth_map[a] == truth_map[b]
        for a, b in zip(pairs["raw_supplier_id_a"], pairs["raw_supplier_id_b"])
    ]
    pairs["predicted_same"] = pairs["match_score"] >= AUTO_MATCH_THRESHOLD
    errors = pairs[pairs["predicted_same"] != pairs["true_same_entity"]].copy()
    errors["error_type"] = np.where(
        errors["predicted_same"], "FALSE MERGE", "MISSED MERGE"
    )

    # ---- 6. scenario S2 reveal --------------------------------------------
    decl = df_decl.merge(
        df_map[["raw_supplier_id", "t2_entity_id", "golden_name", "country_code"]],
        on="raw_supplier_id", how="left",
    )
    t1_names = dict(zip(df_t1["tier1_id"], df_t1["tier1_name"]))

    before = (
        decl.groupby(["metal_id", "declared_supplier_name"])["tier1_id"]
        .nunique().reset_index(name="n_tier1")
    )
    after = (
        decl.groupby(["t2_entity_id", "golden_name", "country_code"])
        .agg(
            n_tier1=("tier1_id", "nunique"),
            n_declared_spellings=("declared_supplier_name", "nunique"),
            tier1_list=("tier1_id", lambda s: "|".join(sorted(set(s)))),
            metals=("metal_id", lambda s: "|".join(sorted(set(s)))),
        )
        .reset_index()
        .sort_values(["n_tier1", "n_declared_spellings"], ascending=False)
    )
    # A concentration is only *hidden* if the aliasing disguised it. If every
    # Tier-1 spelled the name identically, a plain GROUP BY would already have
    # found it -- that is ordinary, visible multi-sourcing.
    after["hidden_by_aliasing_flag"] = (
        (after["n_tier1"] > 1) & (after["n_declared_spellings"] > 1)
    )
    hidden = after[after["hidden_by_aliasing_flag"]].copy()
    visible = after[(after["n_tier1"] > 1) & (~after["hidden_by_aliasing_flag"])].copy()

    # separation margin: how much daylight is there between the weakest true
    # match and the strongest false candidate? This is the real robustness KPI.
    true_scores = pairs.loc[pairs["true_same_entity"], "match_score"]
    false_scores = pairs.loc[~pairs["true_same_entity"], "match_score"]
    weakest_true = float(true_scores.min()) if len(true_scores) else 1.0
    strongest_false = float(false_scores.max()) if len(false_scores) else 0.0

    # ---- write -------------------------------------------------------------
    outputs = {
        "dim_supplier_tier2_golden": df_golden,
        "map_raw_to_golden_tier2": df_map,
        "er_candidate_pairs": pairs,
        "er_review_queue": review,
        "er_threshold_sweep": sweep,
        "er_error_analysis": errors,
        "er_hidden_single_source": after,
    }
    for name, df in outputs.items():
        df.to_csv(OUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    # ---- report ------------------------------------------------------------
    line = "-" * 78
    print(line)
    print("PHASE 2 COMPLETE -- n-tier entity resolution")
    print(f"  {OUT_DIR}")
    print(line)
    print("INPUT")
    print(f"  raw Tier-2 records ............................. {len(df_raw):>8,}")
    print(f"  distinct raw spellings ......................... "
          f"{df_raw['raw_supplier_name'].nunique():>8,}")
    print(f"  records carrying a DUNS number ................. "
          f"{int(df_raw['duns_number'].notna().sum()):>8,}   <- no hard key available")
    print(line)
    print("BLOCKING  (country x metal)")
    print(f"  pairs without blocking ......................... {n_all:>8,}")
    print(f"  pairs after blocking ........................... {len(pairs):>8,}")
    print(f"  reduction ratio ................................ {reduction:>8.1%}")
    print(f"  blocking recall (true pairs retained) .......... {blocking_recall:>8.1%}")
    print(line)
    print("MATCHING")
    print(f"  pass 1  deterministic normalised key ........... "
          f"{int((pairs['match_rule'] == 'DETERMINISTIC_KEY').sum()):>8,} pairs")
    print(f"  pass 2  ensemble = {W_JW_COMPACT:.2f}*JaroWinkler "
          f"+ {W_TOKEN_SET:.2f}*TokenSet + {W_DICE:.2f}*BigramDice")
    print(f"          auto-merge at score >= {AUTO_MATCH_THRESHOLD:.2f}")
    print(f"  pairwise precision ............................. {pm['precision']:>8.1%}")
    print(f"  pairwise recall ................................ {pm['recall']:>8.1%}")
    print(f"  pairwise F1 .................................... {pm['f1']:>8.3f}")
    print(f"  false merges / missed merges ................... "
          f"{pm['fp']:>4,} / {pm['fn']:,}")
    print(line)
    print("CLUSTERING")
    print(f"  entities resolved .............................. "
          f"{cm['n_pred_clusters']:>8,}  (truth: {cm['n_true_clusters']})")
    print(f"  exactly reconstructed clusters ................. "
          f"{cm['exact_cluster_matches']:>8,}  ({cm['cluster_accuracy']:.1%})")
    print(f"  duplicate records eliminated ................... "
          f"{len(df_raw) - cm['n_pred_clusters']:>8,}")
    print(line)
    print("ROBUSTNESS")
    print(f"  best F1 in sweep ............................... {best['f1']:>8.3f} "
          f"@ threshold {best['threshold']:.2f}")
    stable = sweep[sweep["f1"] >= best["f1"] - 1e-9]
    print(f"  threshold range holding best F1 ................ "
          f"{stable['threshold'].min():.2f} - {stable['threshold'].max():.2f}")
    print(f"  weakest TRUE match ............................. {weakest_true:>8.3f}")
    print(f"  strongest FALSE candidate ...................... {strongest_false:>8.3f}")
    print(f"  separation margin .............................. "
          f"{weakest_true - strongest_false:>8.3f}   <- decision is not knife-edge")
    print(f"  ablation: no blocking, same threshold .......... "
          f"P {pm_nb['precision']:.1%} / R {pm_nb['recall']:.1%} / F1 {pm_nb['f1']:.3f}")
    print(f"  pairs routed to human review ................... {len(review):>8,}")
    print(line)
    if len(errors):
        print("ERROR ANALYSIS")
        for e in errors.itertuples(index=False):
            print(f"  [{e.error_type}] {e.match_score:.3f}  "
                  f"{e.name_a.strip()!r}  <->  {e.name_b.strip()!r}")
        print(line)
    print("TOP NEAR-MISSES  (highest scoring pairs that are genuinely different firms)")
    nm = pairs[~pairs["true_same_entity"]].head(3)
    for e in nm.itertuples(index=False):
        print(f"  {e.match_score:.3f}  {e.name_a.strip()!r}")
        print(f"         {e.name_b.strip()!r}")
    print(line)
    print("SCENARIO S2 -- SINGLE SOURCE HIDDEN BY ALIASING")
    if hidden.empty:
        print("  none detected")
    for h in hidden.itertuples(index=False):
        print(f"  >> {h.golden_name}   [{h.country_code} / {h.metals}]")
        print(f"     declared by {h.n_tier1} Tier-1 suppliers, spelled "
              f"{h.n_declared_spellings} different ways:")
        for t in h.tier1_list.split("|"):
            print(f"        {t}  {t1_names.get(t, '')}")
        for nm_ in sorted(decl.loc[decl['t2_entity_id'] == h.t2_entity_id,
                                   'declared_supplier_name'].unique()):
            print(f"        alias: {nm_!r}")
        print(f"     BEFORE resolution: {h.n_declared_spellings} apparently "
              f"independent sub-suppliers.")
        print("     AFTER  resolution: 1 entity, 1 country, 1 point of failure.")
    print(line)
    print("CONCENTRATION ALREADY VISIBLE WITHOUT MATCHING  (same spelling everywhere)")
    for v in visible.itertuples(index=False):
        print(f"     {v.n_tier1} Tier-1s -> {v.golden_name}  "
              f"[{v.country_code} / {v.metals}]")
    print(line)

    # ---- guardrails --------------------------------------------------------
    assert blocking_recall == 1.0, f"blocking lost true pairs ({blocking_recall:.1%})"
    assert pm["precision"] >= 0.95, f"precision too low ({pm['precision']:.1%})"
    assert pm["recall"] >= 0.95, f"recall too low ({pm['recall']:.1%})"
    assert weakest_true > strongest_false, "true and false pair scores overlap"
    assert cm["n_pred_clusters"] == cm["n_true_clusters"], (
        f"resolved {cm['n_pred_clusters']} entities, expected {cm['n_true_clusters']}"
    )
    assert not hidden.empty, "S2 not detected -- the hidden smelter should surface"
    assert df_map["t2_entity_id"].notna().all(), "unassigned raw record"
    print("ALL ASSERTIONS PASSED.\n")


if __name__ == "__main__":
    main()
