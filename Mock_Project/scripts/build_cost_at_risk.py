"""
Global 2nd-Tier Metals Transparency & Cost-at-Risk Engine
=========================================================
PHASE 2b -- the transform pipeline (gold layer + cost-at-risk engine).

Business problem
----------------
Phase 2a resolved *who* the 2nd-tier network really is. That answers the
transparency question but not the commercial one. A purchase order records
what was paid; it does not record what *should* have been paid, which part
of the increase the index actually justifies, or what the contract book
will cost the moment its price protection expires.

This script turns the raw star schema plus the resolved entities into a
gold layer that a semantic model can sit on directly -- one row per
business fact, every measure additive, no analytics hidden in DAX.

What this script does
---------------------
1. PRICE BOOK    index x FX -> one EUR/kg series per metal per month
2. EXPOSURE      PO x BOM   -> embedded metal kg and EUR per PO line
3. SHOULD-COST   replays each contract clause against the published index,
                 calibrates the supplier's base price on its own opening
                 months, and splits actual-vs-should into metal-justified,
                 inflation-justified and UNEXPLAINED (scenario S3)
4. COST-AT-RISK  repricing shock on fixed-price expiry (S1), asymmetric
                 clause leakage (S7) and reset-lag cost, per contract
5. ALLOCATION    Tier-1 metal value pushed down to resolved Tier-2
                 entities via declared shares, with a coverage flag (S6)
6. RISK SCORE    composite Tier-2 score from concentration, cyber (S4),
                 geopolitics and compliance, weighted by money at stake
7. KPI           one tall table of executive measures, scenario-tagged

Design notes
------------
* Pandas only, but written as set-based joins and vectorised columns so
  every step maps 1:1 onto PySpark for Microsoft Fabric (no iterrows, no
  Python-side row loops in the fact builders).
* Every derived column is reproducible from published inputs -- contract
  terms, exchange settlements and FX. Nothing is calibrated on the answer.
* The script ends with self-auditing assertions, the same guardrail
  pattern used by Phase 1 and Phase 2a.

Inputs   : Mock_Project/data/raw/*.csv
           Mock_Project/data/processed/map_raw_to_golden_tier2.csv
           Mock_Project/data/processed/dim_supplier_tier2_golden.csv
Outputs  : Mock_Project/data/gold/*.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Supplier names carry Turkish, German and Polish characters. On Windows a
# redirected stdout defaults to cp1252 and would kill the run on a print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
OUT_DIR = BASE_DIR / "data" / "gold"

# Published planning assumption for non-metal cost escalation (labour,
# energy, logistics, tooling). This is the figure a controlling function
# signs off on; it is NOT fitted to the observed prices.
NON_METAL_ESCALATION_PA = 0.018

# Months of a contract's own history used to calibrate its base price.
CALIBRATION_MONTHS = 6

# A variance is only worth a price letter if it clears both a percentage
# and an absolute materiality floor.
UNEXPLAINED_PCT_THRESHOLD = 0.02
UNEXPLAINED_EUR_THRESHOLD = 25_000.0

# Composite Tier-2 risk score weights (sum to 1.0).
W_CONCENTRATION = 0.30
W_CYBER = 0.25
W_GEOPOLITICAL = 0.25
W_COMPLIANCE = 0.20

RISK_TIER_BOUNDS = (60.0, 40.0)  # >= 60 critical, >= 40 elevated, else managed

LKSG_RISK_WEIGHT = {"High": 1.0, "Medium": 0.5, "Low": 0.2}

# CBAM applies to goods *imported into* the customs union, so the origin of
# the producing entity decides whether a certificate is owed.
EU_MEMBER_STATES = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR",
    "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK",
    "SI", "ES", "SE",
}


def load(name: str, directory: Path) -> pd.DataFrame:
    return pd.read_csv(directory / f"{name}.csv")


# ---------------------------------------------------------------------------
# 1. Price book -- one EUR/kg series per metal per month
# ---------------------------------------------------------------------------

def build_price_book(idx: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """Index settlements are quoted per tonne in the exchange currency.

    The buyer pays in EUR per kg, so both a currency and a unit conversion
    stand between the screen price and the P&L. Keeping them in one table
    means every downstream measure uses the same number.
    """
    px = idx.merge(
        fx[["year_month", "currency_code", "eur_per_unit"]],
        left_on=["year_month", "quote_currency"],
        right_on=["year_month", "currency_code"],
        how="left",
    )
    assert px["eur_per_unit"].notna().all(), "missing FX rate for an index month"

    px["price_eur_per_t"] = px["settlement_price"] * px["eur_per_unit"]
    px["eur_per_kg"] = px["price_eur_per_t"] / 1000.0

    base = px[px["month_index"] == px["month_index"].min()][["metal_id", "eur_per_kg"]]
    base = base.rename(columns={"eur_per_kg": "base_eur_per_kg"})
    px = px.merge(base, on="metal_id", how="left")

    # Index move relative to the first observed month -- the single number
    # every clause in the contract book is written against.
    px["index_move_vs_base"] = px["eur_per_kg"] / px["base_eur_per_kg"] - 1.0

    # Decompose the move into the part the metal did and the part the
    # currency did. For a EUR buyer these are two different negotiations.
    px["idx_move_quote_ccy"] = px.groupby("metal_id")["settlement_price"].transform(
        lambda s: s / s.iloc[0] - 1.0
    )
    px["fx_amplification"] = (
        (1 + px["index_move_vs_base"]) / (1 + px["idx_move_quote_ccy"]) - 1.0
    )

    return px[
        [
            "metal_id", "year_month", "month_index", "quote_currency",
            "settlement_price", "eur_per_unit", "price_eur_per_t", "eur_per_kg",
            "base_eur_per_kg", "index_move_vs_base", "idx_move_quote_ccy",
            "fx_amplification",
        ]
    ]


# ---------------------------------------------------------------------------
# 2. fact_metal_exposure -- PO line x metal
# ---------------------------------------------------------------------------

def build_metal_exposure(
    po: pd.DataFrame, bom: pd.DataFrame, price: pd.DataFrame
) -> pd.DataFrame:
    """Convert 'what we paid a supplier' into 'what metal we bought'.

    This is the hinge of the whole model: spend by supplier is a purchasing
    view, spend by embedded metal is a commodity-management view. Only the
    second one can be hedged, indexed or negotiated.
    """
    exp = po.merge(
        bom[["part_id", "metal_id", "effective_kg_per_unit", "cost_share"]],
        on="part_id",
        how="inner",
    )

    exp["metal_kg"] = exp["order_qty"] * exp["effective_kg_per_unit"]
    # Metal value as billed: the share of the invoice attributable to metal.
    exp["metal_spend_eur"] = exp["net_value_eur"] * exp["cost_share"]

    exp = exp.merge(
        price[["metal_id", "year_month", "eur_per_kg", "index_move_vs_base"]],
        on=["metal_id", "year_month"],
        how="left",
    )
    assert exp["eur_per_kg"].notna().all(), "PO month missing from the price book"

    # Metal value at the spot index in the month of order. The difference
    # against the billed value is the contract's shielding effect: positive
    # means the clause protected us, negative means we overpaid the market.
    exp["metal_value_at_index_eur"] = exp["metal_kg"] * exp["eur_per_kg"]
    exp["index_vs_billed_delta_eur"] = (
        exp["metal_value_at_index_eur"] - exp["metal_spend_eur"]
    )

    return exp[
        [
            "po_id", "po_date", "year_month", "part_id", "tier1_id", "contract_id",
            "metal_id", "order_qty", "net_value_eur", "cost_share", "metal_kg",
            "metal_spend_eur", "eur_per_kg", "index_move_vs_base",
            "metal_value_at_index_eur", "index_vs_billed_delta_eur",
        ]
    ]


# ---------------------------------------------------------------------------
# 3. fact_should_cost -- PO line
# ---------------------------------------------------------------------------

def _reset_month_index(po_m: pd.Series, freq: pd.Series, lag: pd.Series,
                       n_months: int) -> pd.Series:
    """Which index month does a clause actually reprice against?

    A quarterly clause with a 3-month lag is not tracking the market; it is
    tracking the market as it was up to six months ago. Getting this wrong
    silently mis-states every should-cost number, so it is modelled
    explicitly rather than assumed to be the current month.
    """
    m = po_m - lag
    m = np.where(freq == "Q", (m // 3) * 3, m)
    m = np.where(freq == "A", (m // 12) * 12, m)
    return pd.Series(np.clip(m, 0, n_months - 1), index=po_m.index)


def build_should_cost(
    po: pd.DataFrame,
    contract: pd.DataFrame,
    bom: pd.DataFrame,
    price: pd.DataFrame,
    dim_part: pd.DataFrame,
    ym_to_m: dict[str, int],
    surcharge_from: dict[str, pd.Timestamp],
) -> pd.DataFrame:
    """Replay every contract clause against the published index.

    should-cost = base price x (1 + clause adjustment) x non-metal escalation

    The clause adjustment is fully determined by terms we signed and prices
    the exchange published, so it is not an opinion. Anything the actual
    price does on top of it is either the supplier's margin moving or a
    surcharge that has not been justified -- which is exactly what a price
    letter has to defend.
    """
    n_months = int(price["month_index"].max()) + 1

    cterms = contract[
        [
            "contract_id", "part_id", "tier1_id", "sourcing_role", "valid_from",
            "valid_to", "metal_clause_type", "index_metal_id", "passthrough_pct",
            "reset_frequency", "reset_lag_months", "cap_pct", "floor_pct",
        ]
    ]

    sc = po.merge(cterms, on=["contract_id", "part_id", "tier1_id"], how="left")
    assert sc["metal_clause_type"].notna().all(), "PO without a contract clause"

    # Cost share of the metal the clause is written against (a contract
    # indexes on one metal even though the part contains several).
    sc = sc.merge(
        bom[["part_id", "metal_id", "cost_share"]].rename(
            columns={"metal_id": "index_metal_id", "cost_share": "indexed_cost_share"}
        ),
        on=["part_id", "index_metal_id"],
        how="left",
    )
    sc["indexed_cost_share"] = sc["indexed_cost_share"].fillna(0.0)

    sc["month_index"] = sc["year_month"].map(ym_to_m)
    sc["reset_month_index"] = _reset_month_index(
        sc["month_index"], sc["reset_frequency"], sc["reset_lag_months"], n_months
    )

    # Index move the clause is allowed to see (lagged), and the move the
    # market actually made (current). The gap between them is the lag cost.
    mv = price[["metal_id", "month_index", "index_move_vs_base"]]
    sc = sc.merge(
        mv.rename(columns={"metal_id": "index_metal_id",
                           "month_index": "reset_month_index",
                           "index_move_vs_base": "index_move_at_reset"}),
        on=["index_metal_id", "reset_month_index"], how="left",
    ).merge(
        mv.rename(columns={"metal_id": "index_metal_id",
                           "index_move_vs_base": "index_move_current"}),
        on=["index_metal_id", "month_index"], how="left",
    )

    # Uncapped pass-through, then the clause as actually written.
    sc["uncapped_adj"] = (
        sc["passthrough_pct"] * sc["indexed_cost_share"] * sc["index_move_at_reset"]
    )
    sc["clause_adj"] = sc["uncapped_adj"].clip(lower=sc["floor_pct"], upper=sc["cap_pct"])

    sc["escalation_factor"] = 1.0 + NON_METAL_ESCALATION_PA * (sc["month_index"] / 12.0)

    # ---- base price calibration -------------------------------------------
    # The supplier's opening price is commercially agreed, not published, so
    # it is recovered from the contract's own first months by stripping out
    # the clause and the escalation that were in force at the time.
    sc["deflated_unit_price"] = sc["unit_price_eur"] / (
        (1 + sc["clause_adj"]) * sc["escalation_factor"]
    )

    first_m = sc.groupby("contract_id")["month_index"].transform("min")
    sc["in_calibration_window"] = sc["month_index"] < first_m + CALIBRATION_MONTHS

    base_px = (
        sc[sc["in_calibration_window"]]
        .groupby("contract_id")["deflated_unit_price"]
        .median()
        .rename("base_price_eur")
    )
    sc = sc.merge(base_px, on="contract_id", how="left")
    assert sc["base_price_eur"].notna().all(), "contract without a calibration window"

    # A contract whose opening months already sit after a supplier's price
    # letter has the surcharge baked into its own baseline. Flag it rather
    # than quietly reporting a clean variance.
    cal_start_date = sc.groupby("contract_id")["po_date"].transform("min")
    letter_date = sc["tier1_id"].map(surcharge_from)
    sc["calibration_confounded_flag"] = letter_date.notna() & (
        cal_start_date >= letter_date
    )

    # ---- the should-cost bridge -------------------------------------------
    sc["should_cost_unit_eur"] = (
        sc["base_price_eur"] * (1 + sc["clause_adj"]) * sc["escalation_factor"]
    )
    sc["should_cost_eur"] = sc["should_cost_unit_eur"] * sc["order_qty"]
    sc["price_variance_eur"] = sc["net_value_eur"] - sc["should_cost_eur"]

    # Decomposition: how much of the price above the opening base price is
    # explained by metal, how much by published inflation, how much by
    # nothing at all. These three add back to the actual price.
    base_value = sc["base_price_eur"] * sc["order_qty"]
    sc["metal_justified_eur"] = base_value * sc["clause_adj"]
    sc["inflation_justified_eur"] = (
        base_value * (1 + sc["clause_adj"]) * (sc["escalation_factor"] - 1)
    )
    sc["unexplained_eur"] = sc["price_variance_eur"]
    sc["unexplained_pct"] = sc["unexplained_eur"] / sc["should_cost_eur"]

    # Lag cost: what the clause would have paid on the current index versus
    # what it paid on the lagged one. Negative means the lag saved money.
    sc["lag_adj_delta"] = (
        (sc["passthrough_pct"] * sc["indexed_cost_share"] * sc["index_move_current"])
        .clip(lower=sc["floor_pct"], upper=sc["cap_pct"])
        - sc["clause_adj"]
    )

    sc["standard_cost_eur"] = sc["part_id"].map(
        dim_part.set_index("part_id")["standard_cost_eur"]
    )
    # Secondary anchor: variance against the cost the business planned with,
    # which carries the supplier's commercial margin as well as the clause.
    sc["should_cost_vs_standard_eur"] = (
        sc["standard_cost_eur"] * (1 + sc["clause_adj"])
        * sc["escalation_factor"] * sc["order_qty"]
    )

    return sc[
        [
            "po_id", "po_date", "year_month", "month_index", "part_id", "tier1_id",
            "contract_id", "metal_clause_type", "index_metal_id", "indexed_cost_share",
            "passthrough_pct", "cap_pct", "floor_pct", "reset_frequency",
            "reset_lag_months", "reset_month_index", "index_move_at_reset",
            "index_move_current", "uncapped_adj", "clause_adj", "lag_adj_delta",
            "escalation_factor", "order_qty", "unit_price_eur", "net_value_eur",
            "base_price_eur", "calibration_confounded_flag", "should_cost_unit_eur",
            "should_cost_eur", "price_variance_eur", "metal_justified_eur",
            "inflation_justified_eur", "unexplained_eur", "unexplained_pct",
            "standard_cost_eur", "should_cost_vs_standard_eur",
        ]
    ]


def build_surcharge_claims(
    should: pd.DataFrame, dim_t1: pd.DataFrame, as_of: pd.Timestamp
) -> pd.DataFrame:
    """Scenario S3 -- test every supplier's claimed increase against the math.

    A price letter is a hypothesis. The BOM, the index and the FX are the
    test. What survives the test is a cost; what does not is a negotiation
    position with a euro value attached.
    """
    s = should.copy()
    s["po_date"] = pd.to_datetime(s["po_date"])

    claims = dim_t1[["tier1_id", "tier1_name", "country_code",
                     "claimed_metal_surcharge_pct", "surcharge_effective_date",
                     "surcharge_justification_text"]].copy()
    claims["claimed_pct"] = claims["claimed_metal_surcharge_pct"].fillna(0) / 100.0
    claims["surcharge_effective_date"] = pd.to_datetime(
        claims["surcharge_effective_date"], errors="coerce"
    )

    rows = []
    for c in claims.itertuples(index=False):
        sup = s[s["tier1_id"] == c.tier1_id]
        if sup.empty:
            continue
        since = c.surcharge_effective_date
        window = sup if pd.isna(since) else sup[sup["po_date"] >= since]
        if window.empty:
            continue

        spend = float(window["net_value_eur"].sum())
        should_eur = float(window["should_cost_eur"].sum())
        unexplained = spend - should_eur
        # Weighted by value, not by PO count: a 20% overcharge on a small
        # part is noise, 2% on the busbar programme is not. The two
        # justifications are kept apart because only one of them is what
        # the supplier actually claimed.
        base_value = float(window["base_price_eur"].mul(window["order_qty"]).sum())
        base_value = max(base_value, 1.0)
        index_pct = float(window["metal_justified_eur"].sum() / base_value)
        inflation_pct = float(window["inflation_justified_eur"].sum() / base_value)
        months = 1 if pd.isna(since) else max(
            (as_of.to_period("M") - since.to_period("M")).n + 1, 1
        )
        annualised = spend / months * 12

        rows.append({
            "tier1_id": c.tier1_id,
            "tier1_name": c.tier1_name,
            "country_code": c.country_code,
            "surcharge_effective_date": since,
            "claimed_surcharge_pct": c.claimed_pct,
            "justified_by_index_pct": index_pct,
            "justified_by_inflation_pct": inflation_pct,
            "unexplained_pct": unexplained / should_eur if should_eur else 0.0,
            "spend_since_letter_eur": spend,
            "should_cost_since_letter_eur": should_eur,
            "unexplained_eur": unexplained,
            "annualised_spend_eur": annualised,
            "annualised_unexplained_eur": (
                annualised * (unexplained / should_eur) if should_eur else 0.0
            ),
            "justification_text": c.surcharge_justification_text,
            "months_since_letter": months,
        })

    out = pd.DataFrame(rows)
    out["material_flag"] = (
        (out["unexplained_pct"] >= UNEXPLAINED_PCT_THRESHOLD)
        & (out["annualised_unexplained_eur"] >= UNEXPLAINED_EUR_THRESHOLD)
    )
    return out.sort_values("annualised_unexplained_eur", ascending=False)


# ---------------------------------------------------------------------------
# 4. fact_cost_at_risk -- contract x metal
# ---------------------------------------------------------------------------

def build_cost_at_risk(
    should: pd.DataFrame,
    contract: pd.DataFrame,
    price: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Cost-at-Risk is arithmetic on contracts already signed, not a forecast.

    Three mechanisms, all of them dated and all of them avoidable:
      S1  repricing shock -- a fixed price expires and resets to spot in one step
      S7  clause leakage  -- a floor stops us clawing back when the index falls
          lag cost        -- a reset lag delays a fall we have already earned
    """
    s = should.copy()
    s["po_date"] = pd.to_datetime(s["po_date"])
    last12 = s[s["po_date"] > as_of - pd.DateOffset(months=12)]

    spend12 = last12.groupby("contract_id")["net_value_eur"].sum().rename("spend_12m_eur")
    leak12 = (
        last12.assign(
            leak=lambda d: (d["uncapped_adj"] - d["clause_adj"])
            * d["base_price_eur"] * d["order_qty"],
            lag=lambda d: d["lag_adj_delta"] * d["base_price_eur"] * d["order_qty"],
        )
        .groupby("contract_id")[["leak", "lag"]]
        .sum()
    )

    latest_m = int(price["month_index"].max())
    latest_move = (
        price[price["month_index"] == latest_m]
        .set_index("metal_id")["index_move_vs_base"]
    )

    car = contract[
        [
            "contract_id", "part_id", "tier1_id", "metal_clause_type", "index_metal_id",
            "passthrough_pct", "cap_pct", "floor_pct", "reset_lag_months", "valid_to",
        ]
    ].copy()
    car = car.merge(spend12, on="contract_id", how="left")
    car = car.merge(leak12, on="contract_id", how="left")
    car[["spend_12m_eur", "leak", "lag"]] = car[["spend_12m_eur", "leak", "lag"]].fillna(0.0)

    share = (
        should.groupby("contract_id")["indexed_cost_share"].max().rename("indexed_cost_share")
    )
    car = car.merge(share, on="contract_id", how="left")
    car["indexed_cost_share"] = car["indexed_cost_share"].fillna(0.0)

    car["index_move_to_date"] = car["index_metal_id"].map(latest_move).fillna(0.0)
    car["valid_to"] = pd.to_datetime(car["valid_to"])
    car["months_to_expiry"] = (
        (car["valid_to"].dt.year - as_of.year) * 12
        + (car["valid_to"].dt.month - as_of.month)
    )
    car["expired_flag"] = car["months_to_expiry"] <= 0

    # S1 -- on expiry a fixed price steps straight onto the current index.
    # Partial pass-through contracts carry the same shock net of what they
    # have already absorbed.
    already_passed = (
        car["passthrough_pct"] * car["indexed_cost_share"] * car["index_move_to_date"]
    ).clip(lower=car["floor_pct"], upper=car["cap_pct"])
    full_exposure = car["indexed_cost_share"] * car["index_move_to_date"]
    car["repricing_gap_pct"] = (full_exposure - already_passed).clip(lower=0.0)
    car["repricing_exposure_eur"] = car["repricing_gap_pct"] * car["spend_12m_eur"]

    # Risk the business is actually carrying this year: exposure that lands
    # within the next twelve months, pro-rated by when it lands.
    horizon = car["months_to_expiry"].clip(lower=0, upper=12)
    car["in_year_weight"] = np.where(car["expired_flag"], 1.0, (12 - horizon) / 12.0)
    car["cost_at_risk_12m_eur"] = car["repricing_exposure_eur"] * car["in_year_weight"]

    car["clause_leakage_eur"] = car["leak"].clip(lower=0.0)
    car["reset_lag_cost_eur"] = car["lag"].clip(lower=0.0)
    car["total_cost_at_risk_eur"] = (
        car["cost_at_risk_12m_eur"] + car["clause_leakage_eur"] + car["reset_lag_cost_eur"]
    )

    car["risk_driver"] = np.select(
        [
            car["repricing_exposure_eur"] >= car[["clause_leakage_eur",
                                                  "reset_lag_cost_eur"]].max(axis=1),
            car["clause_leakage_eur"] >= car["reset_lag_cost_eur"],
        ],
        ["S1_REPRICING_ON_EXPIRY", "S7_CLAUSE_LEAKAGE"],
        default="RESET_LAG",
    )
    car.loc[car["total_cost_at_risk_eur"] <= 0, "risk_driver"] = "NONE"

    # The recommendation is the point of the table, so it is a column and
    # not a slide.
    car["mitigation"] = np.select(
        [
            (car["risk_driver"] == "S1_REPRICING_ON_EXPIRY") & (car["months_to_expiry"] <= 6),
            car["risk_driver"] == "S1_REPRICING_ON_EXPIRY",
            car["risk_driver"] == "S7_CLAUSE_LEAKAGE",
            car["risk_driver"] == "RESET_LAG",
        ],
        [
            "Hedge or pre-buy before expiry; open renewal now",
            "Convert to symmetric index clause at renewal",
            "Replace floor with symmetric pass-through",
            "Shorten reset lag to 1 month",
        ],
        default="No action",
    )

    return car.drop(columns=["leak", "lag"]).sort_values(
        "total_cost_at_risk_eur", ascending=False
    )


# ---------------------------------------------------------------------------
# 5. fact_tier2_allocation -- Tier-1 x Tier-2 entity x metal
# ---------------------------------------------------------------------------

def build_tier2_allocation(
    exposure: pd.DataFrame, declared: pd.DataFrame, mapping: pd.DataFrame,
    dim_metal: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Push Tier-1 metal value down to the entities that actually produce it.

    Declared shares are self-reported and unaudited, so the allocation is an
    estimate -- but it is the only estimate that survives the aliasing,
    because it is keyed on the resolved entity and not on the spelling.

    Kilograms are allocated alongside euros: a carbon liability is owed on
    mass, not on money, so CBAM cannot be computed from spend alone.
    """
    t1_metal = (
        exposure.groupby(["tier1_id", "metal_id"])
        .agg(tier1_metal_spend_eur=("metal_spend_eur", "sum"),
             tier1_metal_kg=("metal_kg", "sum"))
        .reset_index()
    )

    alloc = declared.merge(t1_metal, on=["tier1_id", "metal_id"], how="left")
    alloc = alloc.merge(
        mapping[["raw_supplier_id", "t2_entity_id", "golden_name", "country_code"]]
        .rename(columns={"country_code": "resolved_country_code"}),
        on="raw_supplier_id", how="left",
    )
    assert alloc["t2_entity_id"].notna().all(), "declared record not resolved to an entity"

    alloc["tier1_metal_spend_eur"] = alloc["tier1_metal_spend_eur"].fillna(0.0)
    alloc["tier1_metal_kg"] = alloc["tier1_metal_kg"].fillna(0.0)
    alloc["allocated_spend_eur"] = (
        alloc["tier1_metal_spend_eur"] * alloc["declared_spend_share"]
    )
    alloc["allocated_metal_kg"] = alloc["tier1_metal_kg"] * alloc["declared_spend_share"]
    alloc = alloc.merge(
        dim_metal[["metal_id", "cbam_in_scope"]], on="metal_id", how="left"
    )
    alloc["cbam_in_scope"] = alloc["cbam_in_scope"].fillna(False).astype(bool)
    # Only imports carry a CBAM certificate; intra-EU production is already
    # inside the ETS and must not be double counted.
    alloc["import_flag"] = ~alloc["resolved_country_code"].isin(EU_MEMBER_STATES)
    alloc["cbam_liable_flag"] = alloc["cbam_in_scope"] & alloc["import_flag"]
    # Self-declared, never audited, and carrying a fifth of the metal value
    # is a different statement from the same share being audited.
    alloc["unverified_flag"] = ~alloc["verified_by_audit"].astype(bool)
    alloc["low_confidence_flag"] = alloc["self_declared_confidence"].eq("Low")

    # S6 -- coverage is a share of value, not a count of suppliers.
    covered = declared[["tier1_id", "metal_id"]].drop_duplicates().assign(is_covered=True)
    cov = t1_metal.merge(covered, on=["tier1_id", "metal_id"], how="left")
    cov["is_covered"] = cov["is_covered"].notna()
    declared_share = (
        alloc.groupby(["tier1_id", "metal_id"])["declared_spend_share"].sum()
        .rename("declared_share_sum").reset_index()
    )
    cov = cov.merge(declared_share, on=["tier1_id", "metal_id"], how="left")
    cov["declared_share_sum"] = cov["declared_share_sum"].fillna(0.0)
    # Shares are rounded to four decimals at source, so a Tier-1 can declare
    # 100.01% of its own sourcing. Coverage is a share of value and cannot
    # exceed 1; leaving it uncapped puts a 100.01% KPI on an executive page
    # and turns rounding noise into negative blind spend.
    cov["over_declared_flag"] = cov["declared_share_sum"] > 1.0 + 1e-9
    cov["effective_share"] = cov["declared_share_sum"].clip(upper=1.0)
    cov["covered_spend_eur"] = cov["tier1_metal_spend_eur"] * cov["effective_share"]
    cov["blind_spend_eur"] = (
        cov["tier1_metal_spend_eur"] - cov["covered_spend_eur"]
    ).clip(lower=0.0)
    cov["coverage_ratio"] = np.where(
        cov["tier1_metal_spend_eur"] > 0,
        cov["covered_spend_eur"] / cov["tier1_metal_spend_eur"], 0.0,
    )

    cols = [
        "declaration_id", "tier1_id", "metal_id", "raw_supplier_id",
        "declared_supplier_name", "t2_entity_id", "golden_name",
        "resolved_country_code", "declared_country_code", "declared_spend_share",
        "tier1_metal_spend_eur", "tier1_metal_kg", "allocated_spend_eur",
        "allocated_metal_kg", "cbam_in_scope", "import_flag", "cbam_liable_flag",
        "source_system", "self_declared_confidence", "verified_by_audit",
        "unverified_flag", "low_confidence_flag",
    ]
    return alloc[cols], cov


def hhi(series: pd.Series) -> float:
    """Herfindahl-Hirschman Index, the measure competition authorities use.

    Above 2,500 counts as highly concentrated. Computing it on raw spellings
    versus on resolved entities shows exactly how much risk the aliasing hid.
    """
    total = series.sum()
    if total <= 0:
        return 0.0
    w = series / total
    return float((w**2).sum() * 10_000)


def build_concentration(alloc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metal, g in alloc.groupby("metal_id"):
        before = hhi(g.groupby("declared_supplier_name")["allocated_spend_eur"].sum())
        after = hhi(g.groupby("t2_entity_id")["allocated_spend_eur"].sum())
        by_entity = g.groupby(["t2_entity_id", "golden_name"])["allocated_spend_eur"].sum()
        top = by_entity.sort_values(ascending=False)
        # Floating-point noise must not read as a negative understatement.
        understatement = after - before
        understatement = 0.0 if abs(understatement) < 1e-6 else understatement
        rows.append({
            "metal_id": metal,
            "allocated_spend_eur": float(g["allocated_spend_eur"].sum()),
            "n_declared_names": int(g["declared_supplier_name"].nunique()),
            "n_resolved_entities": int(g["t2_entity_id"].nunique()),
            "hhi_as_declared": before,
            "hhi_after_resolution": after,
            "hhi_understatement": understatement,
            "highly_concentrated_flag": after >= 2500,
            "largest_entity": top.index[0][1] if len(top) else None,
            "largest_entity_spend_eur": float(top.iloc[0]) if len(top) else 0.0,
            "largest_entity_share": (
                float(top.iloc[0] / g["allocated_spend_eur"].sum())
                if g["allocated_spend_eur"].sum() > 0 else 0.0
            ),
        })
    return pd.DataFrame(rows).sort_values("allocated_spend_eur", ascending=False)


# ---------------------------------------------------------------------------
# 6. dim_supplier_tier2_risk -- composite score
# ---------------------------------------------------------------------------

def build_tier2_risk(
    golden: pd.DataFrame,
    alloc: pd.DataFrame,
    mapping: pd.DataFrame,
    compliance: pd.DataFrame,
    country: pd.DataFrame,
    disruptions: pd.DataFrame,
) -> pd.DataFrame:
    """One score per resolved entity, weighted by the money behind it.

    Cyber maturity, geopolitics and compliance are only meaningful next to
    exposure. A weak supplier you barely buy from is a nuisance; a weak
    supplier carrying a fifth of your metal value is business continuity.
    """
    risk = golden[
        ["t2_entity_id", "golden_name", "country_code", "supplied_metal_id",
         "tier2_role", "n_raw_variants", "source_systems"]
    ].copy()

    spend = alloc.groupby("t2_entity_id")["allocated_spend_eur"].sum()
    deps = alloc.groupby("t2_entity_id")["tier1_id"].nunique()
    aliases = alloc.groupby("t2_entity_id")["declared_supplier_name"].nunique()
    unver = alloc[alloc["unverified_flag"]].groupby("t2_entity_id")["allocated_spend_eur"].sum()

    risk["allocated_spend_eur"] = risk["t2_entity_id"].map(spend).fillna(0.0)
    risk["n_tier1_dependent"] = risk["t2_entity_id"].map(deps).fillna(0).astype(int)
    risk["n_declared_aliases"] = risk["t2_entity_id"].map(aliases).fillna(0).astype(int)
    risk["unverified_spend_eur"] = risk["t2_entity_id"].map(unver).fillna(0.0)
    total_alloc = max(risk["allocated_spend_eur"].sum(), 1.0)
    risk["share_of_allocated_spend"] = risk["allocated_spend_eur"] / total_alloc
    risk["hidden_by_aliasing_flag"] = (
        (risk["n_tier1_dependent"] > 1) & (risk["n_declared_aliases"] > 1)
    )

    kg = alloc.groupby("t2_entity_id")["allocated_metal_kg"].sum()
    liable = (
        alloc[alloc["cbam_liable_flag"]]
        .groupby("t2_entity_id")["allocated_metal_kg"].sum()
    )
    risk["allocated_metal_kg"] = risk["t2_entity_id"].map(kg).fillna(0.0)
    risk["cbam_liable_metal_kg"] = risk["t2_entity_id"].map(liable).fillna(0.0)
    risk["eu_member_flag"] = risk["country_code"].isin(EU_MEMBER_STATES)
    risk["import_flag"] = ~risk["eu_member_flag"]

    # ---- cyber (S4) -------------------------------------------------------
    comp = compliance[compliance["supplier_tier"] == "Tier-2"].merge(
        mapping[["raw_supplier_id", "t2_entity_id"]],
        left_on="supplier_ref_id", right_on="raw_supplier_id", how="left",
    )
    agg = comp.groupby("t2_entity_id").agg(
        cyber_maturity_score=("cyber_maturity_score", "min"),
        iso27001_flag=("iso27001_flag", "max"),
        secure_data_exchange_flag=("secure_data_exchange_flag", "max"),
        nis2_in_scope=("nis2_in_scope", "max"),
        cyber_incidents_24m=("cyber_incidents_24m", "max"),
        audit_overdue_flag=("audit_overdue_flag", "max"),
        lksg_risk_class=("lksg_risk_class", "first"),
        co2e_kg_per_kg_metal=("co2e_kg_per_kg_metal", "mean"),
        renewable_energy_pct=("renewable_energy_pct", "mean"),
        cbam_relevant=("cbam_relevant", "max"),
        conflict_minerals_declared=("conflict_minerals_declared", "min"),
    )
    risk = risk.merge(agg, on="t2_entity_id", how="left")

    # ---- country and disruption history -----------------------------------
    risk = risk.merge(
        country[["country_code", "country_name", "region", "geopolitical_risk_index",
                 "logistics_risk_index", "export_control_flag",
                 "resource_nationalism_score"]],
        on="country_code", how="left",
    )
    dis = disruptions.groupby(["country_code", "metal_id"]).agg(
        n_disruptions_36m=("event_id", "count"),
        max_severity_score=("severity_score", "max"),
        max_lead_time_impact_days=("lead_time_impact_days", "max"),
    ).reset_index()
    risk = risk.merge(
        dis, left_on=["country_code", "supplied_metal_id"],
        right_on=["country_code", "metal_id"], how="left",
    ).drop(columns=["metal_id"])
    risk[["n_disruptions_36m", "max_severity_score", "max_lead_time_impact_days"]] = risk[
        ["n_disruptions_36m", "max_severity_score", "max_lead_time_impact_days"]
    ].fillna(0)

    # ---- components, each scaled 0..1 -------------------------------------
    max_share = max(risk["share_of_allocated_spend"].max(), 1e-9)
    max_deps = max(risk["n_tier1_dependent"].max(), 1)
    risk["concentration_component"] = (
        0.5 * (risk["share_of_allocated_spend"] / max_share)
        + 0.3 * (risk["n_tier1_dependent"] / max_deps)
        + 0.2 * risk["hidden_by_aliasing_flag"].astype(float)
    ).clip(0, 1)

    maturity = risk["cyber_maturity_score"].fillna(0.0)  # no assessment = worst case
    risk["cyber_component"] = (
        0.50 * ((100 - maturity) / 100).clip(0, 1)
        + 0.20 * (~risk["iso27001_flag"].fillna(False).astype(bool)).astype(float)
        + 0.15 * (risk["cyber_incidents_24m"].fillna(0) / 3).clip(0, 1)
        + 0.15 * (~risk["secure_data_exchange_flag"].fillna(False).astype(bool)).astype(float)
    ).clip(0, 1)

    risk["geopolitical_component"] = (
        0.40 * (risk["geopolitical_risk_index"].fillna(5) / 10).clip(0, 1)
        + 0.20 * (risk["logistics_risk_index"].fillna(5) / 10).clip(0, 1)
        + 0.20 * (risk["resource_nationalism_score"].fillna(5) / 10).clip(0, 1)
        + 0.10 * risk["export_control_flag"].fillna(False).astype(bool).astype(float)
        + 0.10 * (risk["n_disruptions_36m"] / 3).clip(0, 1)
    ).clip(0, 1)

    risk["compliance_component"] = (
        0.40 * risk["lksg_risk_class"].map(LKSG_RISK_WEIGHT).fillna(1.0)
        + 0.25 * risk["audit_overdue_flag"].fillna(True).astype(bool).astype(float)
        + 0.20 * (~risk["conflict_minerals_declared"].fillna(False).astype(bool)).astype(float)
        + 0.15 * (1 - (risk["renewable_energy_pct"].fillna(0) / 100)).clip(0, 1)
    ).clip(0, 1)

    risk["composite_risk_score"] = 100 * (
        W_CONCENTRATION * risk["concentration_component"]
        + W_CYBER * risk["cyber_component"]
        + W_GEOPOLITICAL * risk["geopolitical_component"]
        + W_COMPLIANCE * risk["compliance_component"]
    )

    hi, mid = RISK_TIER_BOUNDS
    risk["risk_tier"] = np.select(
        [risk["composite_risk_score"] >= hi, risk["composite_risk_score"] >= mid],
        ["Critical", "Elevated"], default="Managed",
    )
    # The number that ranks the action list: a bad score only matters in
    # proportion to the value routed through it.
    risk["spend_weighted_risk_eur"] = (
        risk["allocated_spend_eur"] * risk["composite_risk_score"] / 100
    )

    risk["primary_risk_driver"] = risk[
        ["concentration_component", "cyber_component",
         "geopolitical_component", "compliance_component"]
    ].idxmax(axis=1).str.replace("_component", "", regex=False).str.upper()

    return risk.sort_values("spend_weighted_risk_eur", ascending=False)


# ---------------------------------------------------------------------------
# 7. KPI table
# ---------------------------------------------------------------------------

def build_kpis(
    po: pd.DataFrame, exposure: pd.DataFrame, should: pd.DataFrame,
    claims: pd.DataFrame, car: pd.DataFrame, cov: pd.DataFrame,
    conc: pd.DataFrame, risk: pd.DataFrame, price: pd.DataFrame, as_of: pd.Timestamp,
) -> pd.DataFrame:
    total_spend = float(po["net_value_eur"].sum())
    total_metal = float(exposure["metal_spend_eur"].sum())
    coverage = float(cov["covered_spend_eur"].sum() / cov["tier1_metal_spend_eur"].sum())
    blind = float(cov["blind_spend_eur"].sum())
    unexplained = float(claims.loc[claims["material_flag"], "annualised_unexplained_eur"].sum())
    repricing = float(car["cost_at_risk_12m_eur"].sum())
    leakage = float(car["clause_leakage_eur"].sum())
    lag_cost = float(car["reset_lag_cost_eur"].sum())
    cyber_at_risk = float(risk.loc[risk["cyber_maturity_score"].fillna(0) < 50,
                                   "allocated_spend_eur"].sum())
    cu = price[price["metal_id"] == "COPPER"]
    fx_amp = float(cu.loc[cu["month_index"].idxmax(), "fx_amplification"])
    addressable = unexplained + repricing + leakage + lag_cost

    k = [
        ("SPEND", "purchasing_spend_eur", total_spend, "EUR", None),
        ("SPEND", "embedded_metal_value_eur", total_metal, "EUR", None),
        ("SPEND", "metal_share_of_spend", total_metal / total_spend, "ratio", None),
        ("VALUE", "unjustified_surcharge_annual_eur", unexplained, "EUR", "S3"),
        ("VALUE", "cost_at_risk_12m_eur", repricing, "EUR", "S1"),
        ("VALUE", "clause_leakage_annual_eur", leakage, "EUR", "S7"),
        ("VALUE", "reset_lag_cost_annual_eur", lag_cost, "EUR", "S7"),
        ("VALUE", "total_addressable_eur", addressable, "EUR", None),
        ("VALUE", "addressable_share_of_spend", addressable / total_spend, "ratio", None),
        ("RISK", "ntier_transparency_coverage", coverage, "ratio", "S6"),
        ("RISK", "blind_metal_spend_eur", blind, "EUR", "S6"),
        ("RISK", "hidden_single_source_entities",
         float(risk["hidden_by_aliasing_flag"].sum()), "count", "S2"),
        ("RISK", "max_tier1_behind_one_entity",
         float(risk["n_tier1_dependent"].max()), "count", "S2"),
        ("RISK", "metal_value_below_cyber_line_eur", cyber_at_risk, "EUR", "S4"),
        ("RISK", "critical_tier2_entities",
         float((risk["risk_tier"] == "Critical").sum()), "count", None),
        ("RISK", "fx_amplification_copper", fx_amp, "ratio", "S5"),
        ("CBAM", "cbam_liable_metal_t",
         float(risk["cbam_liable_metal_kg"].sum() / 1000), "tonnes", None),
        ("CBAM", "cbam_embedded_emissions_t",
         float((risk["cbam_liable_metal_kg"]
                * risk["co2e_kg_per_kg_metal"].fillna(0)).sum() / 1000),
         "tCO2e", None),
        ("CBAM", "imported_share_of_allocated_spend",
         float(risk.loc[risk["import_flag"], "allocated_spend_eur"].sum()
               / max(risk["allocated_spend_eur"].sum(), 1.0)), "ratio", None),
        ("QUALITY", "po_lines_modelled", float(len(should)), "count", None),
        ("QUALITY", "tier2_entities_scored", float(len(risk)), "count", None),
        ("QUALITY", "calibration_confounded_po_lines",
         float(should["calibration_confounded_flag"].sum()), "count", None),
    ]
    for m in conf_rows(conc):
        k.append(m)

    return pd.DataFrame(k, columns=["kpi_group", "kpi_name", "kpi_value",
                                    "unit", "scenario"]).assign(as_of_date=as_of.date())


def conf_rows(conc: pd.DataFrame) -> list[tuple]:
    out = []
    for r in conc.itertuples(index=False):
        out.append(("RISK", f"hhi_after_resolution_{r.metal_id.lower()}",
                    r.hhi_after_resolution, "index", "S2"))
        out.append(("RISK", f"hhi_understatement_{r.metal_id.lower()}",
                    r.hhi_understatement, "index", "S2"))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    po = load("fact_purchase_orders", RAW_DIR)
    po["po_date"] = pd.to_datetime(po["po_date"])
    bom = load("bom_metal_content", RAW_DIR)
    contract = load("dim_contract", RAW_DIR)
    dim_part = load("dim_part", RAW_DIR)
    dim_t1 = load("dim_supplier_tier1", RAW_DIR)
    dim_metal = load("dim_metal", RAW_DIR)
    idx = load("fact_market_index", RAW_DIR)
    fx = load("fact_fx_rate", RAW_DIR)
    declared = load("map_tier1_tier2_declared", RAW_DIR)
    compliance = load("fact_supplier_compliance", RAW_DIR)
    country = load("dim_country_risk", RAW_DIR)
    disruptions = load("fact_disruption_events", RAW_DIR)

    mapping = load("map_raw_to_golden_tier2", PROC_DIR)
    golden = load("dim_supplier_tier2_golden", PROC_DIR)

    as_of = po["po_date"].max()
    surcharge_from = (
        dim_t1.assign(d=pd.to_datetime(dim_t1["surcharge_effective_date"], errors="coerce"))
        .dropna(subset=["d"]).set_index("tier1_id")["d"].to_dict()
    )

    price = build_price_book(idx, fx)
    ym_to_m = dict(zip(price["year_month"], price["month_index"]))

    exposure = build_metal_exposure(po, bom, price)
    should = build_should_cost(po, contract, bom, price, dim_part, ym_to_m, surcharge_from)
    claims = build_surcharge_claims(should, dim_t1, as_of)
    car = build_cost_at_risk(should, contract, price, as_of)
    alloc, cov = build_tier2_allocation(exposure, declared, mapping, dim_metal)
    conc = build_concentration(alloc)
    risk = build_tier2_risk(golden, alloc, mapping, compliance, country, disruptions)
    kpi = build_kpis(po, exposure, should, claims, car, cov, conc, risk, price, as_of)

    outputs = {
        "dim_price_book": price,
        "fact_metal_exposure": exposure,
        "fact_should_cost": should,
        "fact_surcharge_claims": claims,
        "fact_cost_at_risk": car,
        "fact_tier2_allocation": alloc,
        "fact_ntier_coverage": cov,
        "fact_metal_concentration": conc,
        "dim_supplier_tier2_risk": risk,
        "kpi_executive_summary": kpi,
    }
    for name, df in outputs.items():
        df.to_csv(OUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    # ---- report ------------------------------------------------------------
    line = "-" * 78
    total_spend = float(po["net_value_eur"].sum())
    total_metal = float(exposure["metal_spend_eur"].sum())
    coverage = float(cov["covered_spend_eur"].sum() / cov["tier1_metal_spend_eur"].sum())
    material = claims[claims["material_flag"]]
    addressable = float(
        material["annualised_unexplained_eur"].sum()
        + car["cost_at_risk_12m_eur"].sum()
        + car["clause_leakage_eur"].sum()
        + car["reset_lag_cost_eur"].sum()
    )

    print(line)
    print("PHASE 2b COMPLETE -- transform pipeline & cost-at-risk engine")
    print(f"  {OUT_DIR}")
    print(line)
    print("GOLD LAYER")
    for name, df in outputs.items():
        print(f"  {name:<34} {len(df):>8,} rows x {df.shape[1]:>3} cols")
    print(line)
    print(f"SCOPE   as of {as_of:%d %b %Y},  {len(po):,} PO lines,  "
          f"{po['part_id'].nunique()} parts,  {po['tier1_id'].nunique()} Tier-1s")
    print(f"  purchasing spend ............................... EUR {total_spend:>16,.0f}")
    print(f"  embedded metal value ........................... EUR {total_metal:>16,.0f}"
          f"   ({total_metal/total_spend:.1%})")
    print(line)
    print("SHOULD-COST BRIDGE  (actual vs. clause replayed on the published index)")
    print(f"  metal-justified increase ....................... EUR "
          f"{should['metal_justified_eur'].sum():>16,.0f}")
    print(f"  inflation-justified increase ................... EUR "
          f"{should['inflation_justified_eur'].sum():>16,.0f}")
    print(f"  UNEXPLAINED variance ........................... EUR "
          f"{should['unexplained_eur'].sum():>16,.0f}")
    print(f"  PO lines on a confounded baseline .............. "
          f"{int(should['calibration_confounded_flag'].sum()):>20,}")
    print(line)
    print("S3 -- SURCHARGE CLAIMS TESTED AGAINST THE MATH")
    if material.empty:
        print("  no claim clears the materiality thresholds")
    for c in material.itertuples(index=False):
        print(f"  >> {c.tier1_name} ({c.country_code})  letter effective "
              f"{c.surcharge_effective_date:%b %Y}")
        print(f"     claimed .................................... {c.claimed_surcharge_pct:>8.2%}")
        print(f"     justified by index x BOM x FX .............. {c.justified_by_index_pct:>8.2%}")
        print(f"     justified by non-metal inflation ........... "
              f"{c.justified_by_inflation_pct:>8.2%}")
        print(f"     UNEXPLAINED ................................ {c.unexplained_pct:>8.2%}")
        print(f"     annualised value of the gap ................ EUR "
              f"{c.annualised_unexplained_eur:>12,.0f}")
        print(f"     supplier's stated reason ................... {c.justification_text!r}")
    print(line)
    print("COST-AT-RISK  (contracts already signed)")
    print(f"  S1 repricing on expiry, next 12m ............... EUR "
          f"{car['cost_at_risk_12m_eur'].sum():>16,.0f}")
    print(f"     contracts expiring within 12 months ......... "
          f"{int(((car['months_to_expiry'] > 0) & (car['months_to_expiry'] <= 12)).sum()):>20,}")
    print(f"  S7 clause leakage, annual ...................... EUR "
          f"{car['clause_leakage_eur'].sum():>16,.0f}")
    print(f"     contracts leaking value ..................... "
          f"{int((car['clause_leakage_eur'] > 1).sum()):>20,}")
    print(f"  reset-lag cost, annual ......................... EUR "
          f"{car['reset_lag_cost_eur'].sum():>16,.0f}")
    print(line)
    print(f"  TOTAL ADDRESSABLE .............................. EUR {addressable:>16,.0f}")
    print(f"  as a share of purchasing spend ................. "
          f"{addressable/total_spend:>19.2%}")
    print(line)
    print("N-TIER TRANSPARENCY")
    print(f"  coverage, weighted by metal value (S6) ......... {coverage:>19.1%}")
    print(f"  metal value with no declared Tier-2 ............ EUR "
          f"{cov['blind_spend_eur'].sum():>16,.0f}")
    print(f"  Tier-1 x metal pairs fully undeclared .......... "
          f"{int((~cov['is_covered']).sum()):>20,}")
    print(line)
    print("CONCENTRATION  (HHI, >2500 = highly concentrated)")
    for c in conc.itertuples(index=False):
        flag = "  <- highly concentrated" if c.highly_concentrated_flag else ""
        print(f"  {c.metal_id:<12} declared {c.hhi_as_declared:>8,.0f}  ->  "
              f"resolved {c.hhi_after_resolution:>8,.0f}   "
              f"(+{c.hhi_understatement:,.0f}){flag}")
        print(f"               largest entity: {c.largest_entity} "
              f"({c.largest_entity_share:.1%} of allocated value)")
    print(line)
    print("TOP TIER-2 ENTITIES BY MONEY-WEIGHTED RISK")
    cols = ["golden_name", "country_code", "composite_risk_score", "risk_tier",
            "allocated_spend_eur", "n_tier1_dependent", "cyber_maturity_score",
            "primary_risk_driver"]
    for r in risk.head(5)[cols].itertuples(index=False):
        print(f"  {r.composite_risk_score:>5.1f}  {r.risk_tier:<9} "
              f"{r.golden_name[:38]:<38} [{r.country_code}]")
        print(f"         EUR {r.allocated_spend_eur:>12,.0f}   "
              f"{r.n_tier1_dependent} Tier-1s   cyber {r.cyber_maturity_score:.0f}   "
              f"driver {r.primary_risk_driver}")
    print(line)

    # ---- guardrails --------------------------------------------------------
    assert len(should) == len(po), "should-cost lost or duplicated PO lines"
    assert should["should_cost_eur"].gt(0).all(), "non-positive should-cost"
    assert exposure["metal_spend_eur"].le(exposure["net_value_eur"] + 1e-6).all(), (
        "embedded metal value exceeds the invoice"
    )
    assert np.allclose(
        should["net_value_eur"],
        should["base_price_eur"] * should["order_qty"]
        + should["metal_justified_eur"] + should["inflation_justified_eur"]
        + should["unexplained_eur"],
        rtol=1e-6,
    ), "price bridge does not reconcile to the invoice"
    assert car["total_cost_at_risk_eur"].ge(0).all(), "negative cost-at-risk"
    assert car["cost_at_risk_12m_eur"].sum() > 0, "S1 not detected -- expiry shock missing"
    assert car["clause_leakage_eur"].sum() > 0, "S7 not detected -- no clause leakage"
    assert not material.empty, "S3 not detected -- the surcharge claim should fail its test"
    assert 0 < coverage < 1, f"S6 implausible -- coverage {coverage:.1%}"
    assert cov["coverage_ratio"].between(0, 1).all(), "coverage ratio outside 0..1"
    assert cov["blind_spend_eur"].ge(0).all(), "negative blind spend"
    assert (conc["hhi_understatement"] >= -1e-9).all(), (
        "resolution reduced concentration, which is impossible"
    )
    assert conc["highly_concentrated_flag"].any(), "S2 not detected -- no concentrated metal"
    assert risk["hidden_by_aliasing_flag"].any(), "S2 not detected -- no aliased entity"
    assert risk["composite_risk_score"].between(0, 100).all(), "risk score out of range"
    assert risk["t2_entity_id"].is_unique, "duplicate entity in the risk dimension"
    assert len(risk) == len(golden), "entities lost between resolution and scoring"
    assert np.isclose(
        alloc["allocated_metal_kg"].sum(),
        (alloc["tier1_metal_kg"] * alloc["declared_spend_share"]).sum(),
    ), "allocated kilograms do not reconcile to the declared shares"
    assert risk["cbam_liable_metal_kg"].sum() > 0, "no CBAM-liable import found"
    assert (
        risk.loc[risk["eu_member_flag"], "cbam_liable_metal_kg"] == 0
    ).all(), "EU-domiciled entity marked CBAM liable"
    assert kpi["kpi_value"].notna().all(), "null KPI"
    print("ALL ASSERTIONS PASSED.\n")


if __name__ == "__main__":
    main()
