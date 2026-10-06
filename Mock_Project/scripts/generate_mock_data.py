"""
Global 2nd-Tier Metals Transparency & Cost-at-Risk Engine
=========================================================
PHASE 1 -- Synthetic data generation.

Generates a realistic, fully relational synthetic dataset for a discrete
electrical-equipment manufacturer (circuit breakers, contactors, switchgear,
busbar systems) sourcing aluminium, copper and steel through a Tier-1 supplier
base with partially transparent Tier-2 (sub-supplier) networks.

Dependencies : pandas, numpy   (nothing else -- fully self-contained)
Reproducible : single seeded RNG, fixed call order, fixed rounding
Output       : Mock_Project/data/raw/*.csv  (UTF-8-sig, comma separated)

Business scenarios deliberately baked into the numbers
------------------------------------------------------
S1  Silent margin bleed .... copper-heavy FIXED_PRICE contracts expiring AFTER
                             the copper ramp -> repricing shock not yet in P&L
S2  Hidden single source ... one Chinese aluminium smelter sits behind three
                             Tier-1s under three different spellings
S3  Unjustified surcharge .. Tier-1 #4 claims +11.5% "metal costs"; the BOM x
                             index math justifies far less -> negotiation lever
S4  Compliance ~ spend ..... the largest Tier-2s have the weakest cyber posture
S5  FX amplification ....... USD strength turns a +38% USD copper move into a
                             materially larger EUR move
S6  Transparency gap ....... ~28% of Tier-1 metal spend has no declared Tier-2
S7  Lag exploitation ....... capped/lagged clauses fail to capture favourable
                             index drops
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
RNG = np.random.default_rng(SEED)

START_DATE = pd.Timestamp("2023-01-01")
END_DATE = pd.Timestamp("2025-12-31")
N_MONTHS = 36
AS_OF = END_DATE  # "today" from the report's point of view

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# S3 -- the claimed metal surcharge of Tier-1 #4
SURCHARGE_TIER1 = "T1-004"
SURCHARGE_PCT = 0.115
SURCHARGE_FROM = pd.Timestamp("2024-07-01")

# S6 -- target share of Tier-1 metal spend without a declared Tier-2
TARGET_UNCOVERED_SHARE = 0.28


def _m(ts: pd.Timestamp) -> int:
    """Month index 0..35 relative to START_DATE."""
    return (ts.year - START_DATE.year) * 12 + (ts.month - START_DATE.month)


MONTH_STARTS = pd.date_range(START_DATE, periods=N_MONTHS, freq="MS")


# ---------------------------------------------------------------------------
# 1. dim_date
# ---------------------------------------------------------------------------

def build_dim_date() -> pd.DataFrame:
    days = pd.date_range(START_DATE, END_DATE, freq="D")
    df = pd.DataFrame({"date_key": days.strftime("%Y%m%d").astype(int), "date": days})
    df["year"] = days.year
    df["quarter"] = "Q" + days.quarter.astype(str)
    df["month"] = days.month
    df["month_name"] = days.strftime("%b")
    df["year_month"] = days.strftime("%Y-%m")
    # Fiscal year runs October -> September
    df["fiscal_year"] = np.where(days.month >= 10, days.year + 1, days.year)
    df["fiscal_quarter"] = "FQ" + (((days.month - 10) % 12) // 3 + 1).astype(str)
    df["is_month_end"] = days.is_month_end
    df["is_working_day"] = days.dayofweek < 5
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


# ---------------------------------------------------------------------------
# 2. dim_metal
# ---------------------------------------------------------------------------

METALS = [
    # metal_id, name, exchange, ccy, base USD or EUR per tonne, sigma, drift, criticality
    ("COPPER", "Copper Grade A Cathode", "LME", "USD", 8500.0, 0.030, 0.0015, "High"),
    ("ALUMINIUM", "Aluminium P1020A Primary", "LME", "USD", 2400.0, 0.032, 0.0010, "High"),
    ("STEEL_HRC", "Steel Hot-Rolled Coil NW-EU", "Platts", "EUR", 750.0, 0.025, -0.0020, "Medium"),
    ("ZINC", "Zinc Special High Grade", "LME", "USD", 2650.0, 0.028, 0.0005, "Medium"),
    ("NICKEL", "Nickel Primary 99.8%", "LME", "USD", 18000.0, 0.040, 0.0000, "High"),
    ("SILVER", "Silver 999 Contact Alloy", "COMEX", "USD", 760000.0, 0.035, 0.0025, "Critical"),
]


def build_dim_metal() -> pd.DataFrame:
    rows = []
    for mid, name, exch, ccy, base, sigma, drift, crit in METALS:
        rows.append(
            {
                "metal_id": mid,
                "metal_name": name,
                "exchange": exch,
                "quote_currency": ccy,
                "quote_unit": "EUR/t" if ccy == "EUR" else "USD/t",
                "base_price_quote_ccy": base,
                "annual_volatility_pct": round(sigma * np.sqrt(12) * 100, 1),
                "criticality": crit,
                "cbam_in_scope": mid in ("ALUMINIUM", "STEEL_HRC"),
                "conflict_mineral_flag": mid in ("SILVER",),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. fact_market_index  (scripted regimes + stochastic noise)
# ---------------------------------------------------------------------------

def _regime(metal_id: str) -> np.ndarray:
    r = np.ones(N_MONTHS)
    if metal_id == "COPPER":
        # flat -> +38% ramp M13..M20 -> partial retrace to +22% by M26 -> hold
        for m in range(N_MONTHS):
            if m < 13:
                r[m] = 1.00
            elif m <= 20:
                r[m] = 1.00 + 0.38 * (m - 12) / 8.0
            elif m <= 26:
                r[m] = 1.38 - 0.16 * (m - 20) / 6.0
            else:
                r[m] = 1.22
    elif metal_id == "ALUMINIUM":
        # demand slump step-down at M8, European energy-crisis spike M24..M28
        for m in range(N_MONTHS):
            if m < 8:
                r[m] = 1.00
            elif m < 24:
                r[m] = 0.88
            elif m <= 28:
                r[m] = 0.88 + 0.30 * (m - 23) / 5.0
            else:
                r[m] = 1.18 - 0.20 * (m - 28) / 7.0
    elif metal_id == "STEEL_HRC":
        r = np.linspace(1.00, 0.86, N_MONTHS)
    elif metal_id == "NICKEL":
        spikes = {15: 1.10, 16: 1.28, 17: 1.55, 18: 1.30, 19: 1.12}
        for m, v in spikes.items():
            r[m] = v
    elif metal_id == "ZINC":
        r = 1.0 + 0.09 * np.sin(np.linspace(0, 2.4 * np.pi, N_MONTHS))
    elif metal_id == "SILVER":
        r = np.linspace(1.00, 1.25, N_MONTHS)
    return r


def build_fact_market_index() -> pd.DataFrame:
    rows = []
    for mid, _, exch, ccy, base, sigma, drift, _ in METALS:
        shocks = RNG.normal(drift, sigma, N_MONTHS)
        walk = np.exp(np.cumsum(shocks))
        path = base * walk * _regime(mid)
        for m in range(N_MONTHS):
            ts = MONTH_STARTS[m]
            rows.append(
                {
                    "metal_id": mid,
                    "year_month": ts.strftime("%Y-%m"),
                    "quote_date": (ts + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d"),
                    "month_index": m,
                    "exchange": exch,
                    "quote_currency": ccy,
                    "settlement_price": round(float(path[m]), 2),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. fact_fx_rate   (S5 -- USD strengthens right through the copper ramp)
# ---------------------------------------------------------------------------

def build_fact_fx_rate() -> pd.DataFrame:
    # EUR received per 1 unit of foreign currency
    usd = np.full(N_MONTHS, 0.920)
    for m in range(N_MONTHS):
        if m < 14:
            usd[m] = 0.920
        elif m <= 22:
            usd[m] = 0.920 * (1 + 0.09 * (m - 13) / 9.0)
        else:
            usd[m] = 1.003 - 0.043 * (m - 22) / 13.0
    usd = usd * np.exp(RNG.normal(0.0, 0.006, N_MONTHS))

    cny = 0.1280 * np.exp(np.cumsum(RNG.normal(-0.0015, 0.008, N_MONTHS)))

    rows = []
    for m in range(N_MONTHS):
        ts = MONTH_STARTS[m]
        ym = ts.strftime("%Y-%m")
        rows.append({"year_month": ym, "month_index": m, "currency_code": "EUR",
                     "eur_per_unit": 1.000000})
        rows.append({"year_month": ym, "month_index": m, "currency_code": "USD",
                     "eur_per_unit": round(float(usd[m]), 6)})
        rows.append({"year_month": ym, "month_index": m, "currency_code": "CNY",
                     "eur_per_unit": round(float(cny[m]), 6)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. dim_supplier_tier1
# ---------------------------------------------------------------------------

TIER1 = [
    # id, name, country, city, families, n_parts, share
    ("T1-001", "Nordmetall Praezision GmbH", "DE", "Hagen",
     ["Busbar System", "Contactor"], 8, 0.19),
    ("T1-002", "Bavaria Contact Systems AG", "DE", "Amberg",
     ["Contactor", "Circuit Breaker"], 8, 0.17),
    ("T1-003", "Lombardia Elettromeccanica S.p.A.", "IT", "Bergamo",
     ["Circuit Breaker"], 6, 0.13),
    ("T1-004", "Adriatic Switchgear Solutions d.o.o.", "SI", "Maribor",
     ["Switchgear"], 7, 0.15),
    ("T1-005", "Anatolia Switchgear Sanayi A.S.", "TR", "Izmir",
     ["Switchgear", "Circuit Breaker"], 6, 0.11),
    ("T1-006", "Shenzhen Precision Components Ltd.", "CN", "Shenzhen",
     ["Contactor", "Circuit Breaker"], 6, 0.10),
    ("T1-007", "Visegrad Elektro Kft.", "HU", "Gyor",
     ["Switchgear", "Busbar System"], 5, 0.09),
    ("T1-008", "Iberia Conductores S.L.", "ES", "Zaragoza",
     ["Busbar System"], 4, 0.06),
]


def build_dim_supplier_tier1() -> pd.DataFrame:
    rows = []
    for sid, name, cc, city, fams, n_parts, share in TIER1:
        claimed = SURCHARGE_PCT if sid == SURCHARGE_TIER1 else 0.0
        rows.append(
            {
                "tier1_id": sid,
                "tier1_name": name,
                "country_code": cc,
                "city": city,
                "product_families": "|".join(fams),
                "payment_terms_days": int(RNG.choice([30, 45, 60, 90])),
                "relationship_since": int(RNG.integers(1998, 2019)),
                "planned_share_of_spend": share,
                "preferred_supplier_flag": bool(share >= 0.13),
                "claimed_metal_surcharge_pct": round(claimed * 100, 2),
                "surcharge_effective_date": (
                    SURCHARGE_FROM.strftime("%Y-%m-%d") if claimed else ""
                ),
                "surcharge_justification_text": (
                    "General increase in metal raw material costs" if claimed else ""
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. dim_supplier_tier2_raw + ground truth
#    20 canonical entities exploded into ~48 dirty name variants
# ---------------------------------------------------------------------------

TIER2_ENTITIES = [
    # (entity_id, canonical_name, country, metal, role, [(variant, source_system), ...])
    ("E001", "Yunnan Hongtai Aluminium Smelting Co., Ltd.", "CN", "ALUMINIUM", "Smelter", [
        ("Yunnan Hongtai Aluminium Smelting Co., Ltd.", "ERP_SAP"),
        ("YUNNAN HONGTAI ALUMINUM SMELTING CO LTD", "SUPPLIER_SURVEY"),
        ("Yunnan Hong Tai Aluminium Smelting", "SUSTAINABILITY_PORTAL"),
        ("Yunnan Hongtai Alu. Smelting Co Ltd.", "ERP_SAP"),
    ]),
    ("E002", "Mueller & Sohn Metallwerke GmbH", "DE", "COPPER", "Rolling Mill", [
        ("Müller & Sohn Metallwerke GmbH", "ERP_SAP"),
        ("Mueller und Sohn Metallwerke G.m.b.H.", "SUPPLIER_SURVEY"),
        ("MULLER & SOHN METALLWERKE GMBH ", "SUSTAINABILITY_PORTAL"),
    ]),
    ("E003", "Nordic Copper Refining AB", "SE", "COPPER", "Refinery", [
        ("Nordic Copper Refining AB", "ERP_SAP"),
        ("Nordic Copper Refining A.B.", "SUPPLIER_SURVEY"),
        ("NORDIC COPPER REFINING", "SUSTAINABILITY_PORTAL"),
    ]),
    ("E004", "Silesia Steel Processing Sp. z o.o.", "PL", "STEEL_HRC", "Service Centre", [
        ("Silesia Steel Processing Sp. z o.o.", "ERP_SAP"),
        ("SILESIA STEEL PROCESSING SP ZOO", "SUPPLIER_SURVEY"),
    ]),
    ("E005", "Anadolu Bakir Sanayi A.S.", "TR", "COPPER", "Rolling Mill", [
        ("Anadolu Bakır Sanayi A.Ş.", "ERP_SAP"),
        ("Anadolu Bakir Sanayi AS", "SUPPLIER_SURVEY"),
        ("ANADOLU BAKIRR SANAYI A.S.", "SUSTAINABILITY_PORTAL"),
    ]),
    ("E006", "Jiangsu Everbright Zinc Alloys Ltd.", "CN", "ZINC", "Smelter", [
        ("Jiangsu Everbright Zinc Alloys Ltd.", "ERP_SAP"),
        ("JIANGSU EVERBRIGHT ZINC ALLOYS CO LTD", "SUPPLIER_SURVEY"),
    ]),
    ("E007", "Rheinland Zinkveredelung GmbH & Co. KG", "DE", "ZINC", "Coater", [
        ("Rheinland Zinkveredelung GmbH & Co. KG", "ERP_SAP"),
        ("Rheinland Zinkveredlung GmbH und Co KG", "SUPPLIER_SURVEY"),
    ]),
    ("E008", "Alpine Aluminium Extrusion AG", "AT", "ALUMINIUM", "Extruder", [
        ("Alpine Aluminium Extrusion AG", "ERP_SAP"),
        ("Alpine Aluminum Extrusion A.G.", "SUPPLIER_SURVEY"),
        ("ALPINE ALUMINIUM EXTRUSION", "SUSTAINABILITY_PORTAL"),
    ]),
    ("E009", "Baltic Nickel Alloys UAB", "LT", "NICKEL", "Alloy Producer", [
        ("Baltic Nickel Alloys UAB", "ERP_SAP"),
        ("BALTIC NICKEL ALLOYS U.A.B.", "SUPPLIER_SURVEY"),
    ]),
    ("E010", "Iberia Silver Contacts S.L.", "ES", "SILVER", "Contact Maker", [
        ("Iberia Silver Contacts S.L.", "ERP_SAP"),
        ("IBERIA SILVER CONTACTS SL", "SUSTAINABILITY_PORTAL"),
    ]),
    ("E011", "Hunan Silver Technology Co., Ltd.", "CN", "SILVER", "Refinery", [
        ("Hunan Silver Technology Co., Ltd.", "ERP_SAP"),
        ("HUNAN SILVER TECHNOLOGY CO LTD", "SUPPLIER_SURVEY"),
    ]),
    ("E012", "Donau Stahlservice Ges.m.b.H.", "AT", "STEEL_HRC", "Service Centre", [
        ("Donau Stahlservice Ges.m.b.H.", "ERP_SAP"),
        ("Donau Stahlservice GesmbH", "SUPPLIER_SURVEY"),
        ("DONAU STAHL SERVICE GMBH", "SUSTAINABILITY_PORTAL"),
    ]),
    ("E013", "Lombardy Metal Forming S.r.l.", "IT", "COPPER", "Former", [
        ("Lombardy Metal Forming S.r.l.", "ERP_SAP"),
        ("LOMBARDY METAL FORMING SRL", "SUPPLIER_SURVEY"),
    ]),
    ("E014", "Emirates Aluminium Trading FZE", "AE", "ALUMINIUM", "Trader", [
        ("Emirates Aluminium Trading FZE", "ERP_SAP"),
        ("Emirates Aluminum Trading F.Z.E.", "SUPPLIER_SURVEY"),
    ]),
    ("E015", "Korea Precision Steel Co., Ltd.", "KR", "STEEL_HRC", "Mill", [
        ("Korea Precision Steel Co., Ltd.", "ERP_SAP"),
        ("KOREA PRECISION STEEL CO LTD", "SUSTAINABILITY_PORTAL"),
    ]),
    ("E016", "Vistula Copper Wire Sp. z o.o.", "PL", "COPPER", "Wire Drawer", [
        ("Vistula Copper Wire Sp. z o.o.", "ERP_SAP"),
        ("Vistula Copper Wire Sp.z.o.o", "SUPPLIER_SURVEY"),
    ]),
    ("E017", "Bohemia Metalltechnik s.r.o.", "CZ", "STEEL_HRC", "Stamper", [
        ("Bohemia Metalltechnik s.r.o.", "ERP_SAP"),
        ("BOHEMIA METALLTECHNIK SRO", "SUPPLIER_SURVEY"),
    ]),
    ("E018", "Andes Copper Cathodes S.A.", "CL", "COPPER", "Mine / Refinery", [
        ("Andes Copper Cathodes S.A.", "ERP_SAP"),
        ("ANDES COPPER CATHODES SA", "SUPPLIER_SURVEY"),
    ]),
    ("E019", "Guangdong Sunrise International Metals Co.", "CN", "ALUMINIUM", "Trader", [
        ("Guangdong Sunrise Int'l Metals Co.", "ERP_SAP"),
        ("Guangdong Sunrise International Metals Company", "SUPPLIER_SURVEY"),
        ("GUANGDONG SUNRISE INTL METALS", "SUSTAINABILITY_PORTAL"),
    ]),
    ("E020", "Saxony Steel & Wire GmbH", "DE", "STEEL_HRC", "Mill", [
        ("Saxony Steel & Wire GmbH", "ERP_SAP"),
        ("Saxony Steel und Wire G.m.b.H.", "SUPPLIER_SURVEY"),
    ]),
]


def build_tier2_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows, truth_rows = [], []
    counter = 1
    for eid, canonical, cc, metal, role, variants in TIER2_ENTITIES:
        for vname, source in variants:
            rid = f"T2R-{counter:03d}"
            counter += 1
            first_seen = START_DATE + pd.Timedelta(days=int(RNG.integers(0, 540)))
            raw_rows.append(
                {
                    "raw_supplier_id": rid,
                    "raw_supplier_name": vname,
                    "country_code": cc,
                    "supplied_metal_id": metal,
                    "tier2_role": role,
                    "source_system": source,
                    "first_seen_date": first_seen.strftime("%Y-%m-%d"),
                    "duns_number": "",  # deliberately missing -> forces name matching
                }
            )
            truth_rows.append(
                {"raw_supplier_id": rid, "true_entity_id": eid, "canonical_name": canonical}
            )
    return pd.DataFrame(raw_rows), pd.DataFrame(truth_rows)


# ---------------------------------------------------------------------------
# 13. dim_country_risk
# ---------------------------------------------------------------------------

COUNTRY_RISK = [
    # code, name, region, geopolitical, export_control, logistics, energy, nationalism
    ("DE", "Germany", "Western Europe", 1.8, False, 2.1, 128.0, 1.0),
    ("IT", "Italy", "Southern Europe", 2.4, False, 3.0, 121.0, 1.5),
    ("ES", "Spain", "Southern Europe", 2.2, False, 2.8, 112.0, 1.4),
    ("AT", "Austria", "Western Europe", 1.7, False, 2.2, 124.0, 1.0),
    ("SI", "Slovenia", "Central Europe", 2.1, False, 2.6, 118.0, 1.2),
    ("HU", "Hungary", "Central Europe", 4.2, False, 3.1, 115.0, 3.4),
    ("PL", "Poland", "Central Europe", 3.1, False, 2.9, 119.0, 2.0),
    ("CZ", "Czechia", "Central Europe", 2.3, False, 2.7, 117.0, 1.3),
    ("SE", "Sweden", "Northern Europe", 1.5, False, 2.4, 96.0, 0.8),
    ("LT", "Lithuania", "Northern Europe", 5.6, False, 3.6, 123.0, 1.9),
    ("TR", "Turkey", "Near East", 6.4, False, 4.2, 134.0, 5.1),
    ("CN", "China", "Asia Pacific", 7.8, True, 5.4, 88.0, 7.2),
    ("KR", "South Korea", "Asia Pacific", 4.8, False, 3.3, 101.0, 2.1),
    ("AE", "United Arab Emirates", "Middle East", 4.1, False, 3.0, 72.0, 3.8),
    ("CL", "Chile", "South America", 3.9, False, 4.6, 105.0, 5.8),
    ("ID", "Indonesia", "Asia Pacific", 6.1, True, 5.1, 84.0, 8.1),
]


def build_dim_country_risk() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "country_code": c,
                "country_name": n,
                "region": r,
                "geopolitical_risk_index": g,
                "export_control_flag": e,
                "logistics_risk_index": lo,
                "energy_price_index": en,
                "resource_nationalism_score": rn,
            }
            for c, n, r, g, e, lo, en, rn in COUNTRY_RISK
        ]
    )


# ---------------------------------------------------------------------------
# 14. fact_disruption_events
# ---------------------------------------------------------------------------

DISRUPTIONS = [
    ("EVT-01", "2023-04-10", "2023-06-18", "CN", "ALUMINIUM", "Energy curtailment Yunnan province", "Medium", 3, 14),
    ("EVT-02", "2023-09-01", "2023-10-20", "DE", "STEEL_HRC", "Rail freight strike Germany", "Low", 2, 7),
    ("EVT-03", "2024-01-15", "2024-05-30", "AE", "ALUMINIUM", "Red Sea rerouting via Cape of Good Hope", "High", 4, 21),
    ("EVT-04", "2024-02-05", "2024-04-12", "ID", "NICKEL", "Export ban on unprocessed ore", "High", 5, 28),
    ("EVT-05", "2024-03-22", "2024-04-30", "CL", "COPPER", "Mine workers strike Atacama region", "High", 4, 18),
    ("EVT-06", "2024-06-01", "2024-08-15", "TR", "COPPER", "Currency crisis, supplier liquidity stress", "Medium", 3, 12),
    ("EVT-07", "2024-09-10", "2024-11-05", "CN", "ALUMINIUM", "Smelter maintenance outage Yunnan Hongtai", "High", 5, 26),
    ("EVT-08", "2024-11-01", "2024-12-20", "PL", "STEEL_HRC", "Service centre fire, capacity loss", "Medium", 3, 15),
    ("EVT-09", "2025-01-12", "2025-03-01", "DE", "ZINC", "Gas price spike, galvanising curtailed", "Medium", 3, 10),
    ("EVT-10", "2025-04-05", "2025-06-10", "CN", "SILVER", "Export licence tightening on precious metals", "High", 4, 22),
    ("EVT-11", "2025-07-01", "2025-08-20", "ES", "ALUMINIUM", "Heatwave production curtailment", "Low", 2, 8),
    ("EVT-12", "2025-09-15", "2025-11-30", "CN", "COPPER", "Port congestion Shenzhen / Yantian", "Medium", 3, 13),
]


def build_fact_disruption_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": e,
                "start_date": s,
                "end_date": en,
                "country_code": c,
                "metal_id": m,
                "event_description": d,
                "severity": sev,
                "severity_score": ss,
                "lead_time_impact_days": lt,
            }
            for e, s, en, c, m, d, sev, ss, lt in DISRUPTIONS
        ]
    )


# ---------------------------------------------------------------------------
# 5 + 6. dim_part and bom_metal_content
# ---------------------------------------------------------------------------

FAMILY_BOM = {
    # family: [(metal, kg_low, kg_high), ...]
    "Circuit Breaker": [("COPPER", 0.55, 2.40), ("STEEL_HRC", 0.80, 3.20),
                        ("SILVER", 0.0008, 0.0045), ("ZINC", 0.05, 0.30)],
    "Contactor": [("COPPER", 0.40, 1.60), ("SILVER", 0.0010, 0.0060),
                  ("STEEL_HRC", 0.35, 1.40), ("NICKEL", 0.004, 0.030)],
    "Switchgear": [("STEEL_HRC", 6.00, 22.00), ("ALUMINIUM", 1.80, 7.50),
                   ("ZINC", 0.20, 0.90), ("COPPER", 0.30, 1.10)],
    "Busbar System": [("COPPER", 3.50, 14.00), ("ALUMINIUM", 0.80, 4.00),
                      ("STEEL_HRC", 0.40, 1.60)],
}

FAMILY_PREFIX = {
    "Circuit Breaker": "CB",
    "Contactor": "CT",
    "Switchgear": "SG",
    "Busbar System": "BB",
}


def build_parts_and_bom(idx_eur_kg: dict[str, np.ndarray]):
    part_rows, bom_rows = [], []
    pid = 1
    for sid, _, _, _, fams, n_parts, _ in TIER1:
        for i in range(n_parts):
            family = fams[i % len(fams)]
            part_id = f"P-{pid:04d}"
            pid += 1
            rating = int(RNG.choice([63, 100, 160, 250, 400, 630, 800, 1250]))
            part_desc = f"{family} {rating}A {FAMILY_PREFIX[family]}-{rating}-{part_id[-3:]}"

            metal_cost_base = 0.0
            for metal, lo, hi in FAMILY_BOM[family]:
                kg = float(RNG.uniform(lo, hi))
                scrap = float(RNG.uniform(0.02, 0.11))
                eff_kg = kg * (1 + scrap)
                metal_cost_base += eff_kg * idx_eur_kg[metal][0]
                bom_rows.append(
                    {
                        "part_id": part_id,
                        "metal_id": metal,
                        "kg_per_unit": round(kg, 4),
                        "scrap_factor": round(scrap, 4),
                        "effective_kg_per_unit": round(eff_kg, 4),
                    }
                )

            metal_ratio = float(RNG.uniform(0.30, 0.58))
            standard_cost = metal_cost_base / metal_ratio
            part_rows.append(
                {
                    "part_id": part_id,
                    "part_description": part_desc,
                    "product_family": family,
                    "current_rating_a": rating,
                    "primary_tier1_id": sid,
                    "annual_volume_units": int(RNG.integers(2_000, 90_000)),
                    "standard_cost_eur": round(standard_cost, 2),
                    "target_gross_margin_pct": round(float(RNG.uniform(22.0, 38.0)), 1),
                    "lifecycle_stage": str(
                        RNG.choice(["Ramp-Up", "Serial", "Serial", "Serial", "Phase-Out"])
                    ),
                }
            )
    return pd.DataFrame(part_rows), pd.DataFrame(bom_rows)


# ---------------------------------------------------------------------------
# 10. dim_contract   (S1 and S7 live here)
# ---------------------------------------------------------------------------

def build_contracts(df_part: pd.DataFrame, bom_cost_share: pd.DataFrame) -> pd.DataFrame:
    dominant = (
        bom_cost_share.sort_values(["part_id", "cost_share"], ascending=[True, False])
        .groupby("part_id", as_index=False)
        .first()[["part_id", "metal_id", "cost_share"]]
        .rename(columns={"metal_id": "dominant_metal_id", "cost_share": "dominant_share"})
    )
    parts = df_part.merge(dominant, on="part_id", how="left")

    rows = []
    cid = 1
    for r in parts.itertuples(index=False):
        # ---- primary contract -------------------------------------------------
        is_copper_led = r.dominant_metal_id == "COPPER"
        if is_copper_led:
            # S1: ~45% of copper-led parts sit on a fixed price that expires
            #     shortly AFTER the end of the observation window
            clause = str(RNG.choice(
                ["FIXED_PRICE", "FULL_INDEX_PASSTHROUGH", "PARTIAL_PASSTHROUGH", "CAPPED_INDEX"],
                p=[0.45, 0.18, 0.22, 0.15],
            ))
        elif r.primary_tier1_id == SURCHARGE_TIER1:
            # S3: the surcharge supplier hides behind fixed prices
            clause = str(RNG.choice(
                ["FIXED_PRICE", "PARTIAL_PASSTHROUGH", "CAPPED_INDEX"], p=[0.7, 0.2, 0.1]))
        else:
            clause = str(RNG.choice(
                ["FIXED_PRICE", "FULL_INDEX_PASSTHROUGH", "PARTIAL_PASSTHROUGH", "CAPPED_INDEX"],
                p=[0.22, 0.30, 0.30, 0.18],
            ))

        if clause == "FIXED_PRICE":
            passthrough, cap, floor = 0.0, 0.0, 0.0
            freq, lag = "A", 0
            # expires just beyond the data horizon -> repricing risk, no spend gap
            valid_to = pd.Timestamp("2026-01-31") + pd.Timedelta(
                days=int(RNG.integers(0, 240)))
        elif clause == "FULL_INDEX_PASSTHROUGH":
            passthrough, cap, floor = 1.00, 1.00, -1.00
            freq = str(RNG.choice(["M", "Q"], p=[0.45, 0.55]))
            lag = int(RNG.integers(1, 3))
            valid_to = pd.Timestamp("2027-12-31")
        elif clause == "PARTIAL_PASSTHROUGH":
            passthrough = round(float(RNG.uniform(0.45, 0.80)), 2)
            cap, floor = 1.00, -1.00
            freq = str(RNG.choice(["Q", "A"], p=[0.7, 0.3]))
            lag = int(RNG.integers(1, 4))
            valid_to = pd.Timestamp("2027-06-30")
        else:  # CAPPED_INDEX -- S7: cap plus long lag eats favourable moves
            passthrough = round(float(RNG.uniform(0.70, 1.00)), 2)
            cap = round(float(RNG.uniform(0.04, 0.09)), 3)
            floor = round(-float(RNG.uniform(0.01, 0.03)), 3)
            freq = "Q"
            lag = 3
            valid_to = pd.Timestamp("2026-09-30")

        rows.append(
            {
                "contract_id": f"C-{cid:04d}",
                "part_id": r.part_id,
                "tier1_id": r.primary_tier1_id,
                "sourcing_role": "Primary",
                "allocation_pct": 100,
                "valid_from": "2023-01-01",
                "valid_to": valid_to.strftime("%Y-%m-%d"),
                "metal_clause_type": clause,
                "index_metal_id": r.dominant_metal_id,
                "index_base_year_month": "2023-01",
                "passthrough_pct": passthrough,
                "reset_frequency": freq,
                "reset_lag_months": lag,
                "cap_pct": cap,
                "floor_pct": floor,
                "incoterm": str(RNG.choice(["DAP", "FCA", "DDP", "EXW"])),
            }
        )
        cid += 1

    # ---- dual-source secondary contracts (10 parts) --------------------------
    dual = parts.sample(n=10, random_state=SEED).sort_values("part_id")
    all_t1 = [t[0] for t in TIER1]
    for r in dual.itertuples(index=False):
        alt = [s for s in all_t1 if s != r.primary_tier1_id]
        rows.append(
            {
                "contract_id": f"C-{cid:04d}",
                "part_id": r.part_id,
                "tier1_id": str(RNG.choice(alt)),
                "sourcing_role": "Secondary",
                "allocation_pct": 30,
                "valid_from": "2023-01-01",
                "valid_to": "2027-12-31",
                "metal_clause_type": "PARTIAL_PASSTHROUGH",
                "index_metal_id": r.dominant_metal_id,
                "index_base_year_month": "2023-01",
                "passthrough_pct": 0.60,
                "reset_frequency": "Q",
                "reset_lag_months": 2,
                "cap_pct": 1.00,
                "floor_pct": -1.00,
                "incoterm": "DAP",
            }
        )
        cid += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 11. fact_purchase_orders
# ---------------------------------------------------------------------------

def _reset_month(po_m: int, freq: str, lag: int) -> int:
    m = po_m - lag
    if freq == "Q":
        m = (m // 3) * 3
    elif freq == "A":
        m = (m // 12) * 12
    return int(np.clip(m, 0, N_MONTHS - 1))


def build_purchase_orders(
    df_contract: pd.DataFrame,
    df_part: pd.DataFrame,
    bom_cost_share: pd.DataFrame,
    idx_eur_kg: dict[str, np.ndarray],
    df_disr: pd.DataFrame,
) -> pd.DataFrame:
    part_cost = df_part.set_index("part_id")["standard_cost_eur"].to_dict()
    part_vol = df_part.set_index("part_id")["annual_volume_units"].to_dict()
    share_map = {
        (r.part_id, r.metal_id): r.cost_share for r in bom_cost_share.itertuples(index=False)
    }
    t1_country = {t[0]: t[2] for t in TIER1}

    disr = df_disr.copy()
    disr["start_date"] = pd.to_datetime(disr["start_date"])
    disr["end_date"] = pd.to_datetime(disr["end_date"])

    rows = []
    po_no = 1
    for c in df_contract.itertuples(index=False):
        v_from = max(pd.Timestamp(c.valid_from), START_DATE)
        v_to = min(pd.Timestamp(c.valid_to), END_DATE)
        if v_to < v_from:
            continue

        base_price = part_cost[c.part_id] * float(RNG.uniform(0.96, 1.06))
        if c.sourcing_role == "Secondary":
            base_price *= float(RNG.uniform(1.02, 1.09))
        metal_share = share_map.get((c.part_id, c.index_metal_id), 0.0)
        idx_base = idx_eur_kg[c.index_metal_id][0]
        monthly_units = part_vol[c.part_id] / 12.0 * (c.allocation_pct / 100.0)
        country = t1_country[c.tier1_id]
        doc_ccy = "CNY" if country == "CN" else "EUR"

        for m in range(_m(v_from), _m(v_to) + 1):
            n_po = int(np.clip(RNG.poisson(5.5), 1, 12))
            reset_m = _reset_month(m, c.reset_frequency, c.reset_lag_months)
            delta = idx_eur_kg[c.index_metal_id][reset_m] / idx_base - 1.0
            adj = float(np.clip(c.passthrough_pct * metal_share * delta, c.floor_pct, c.cap_pct))
            escalation = 1.0 + 0.018 * (m / 12.0)  # non-metal cost inflation

            for _ in range(n_po):
                day = int(RNG.integers(1, 28))
                po_date = MONTH_STARTS[m] + pd.Timedelta(days=day - 1)
                if po_date < v_from or po_date > v_to:
                    continue

                price = base_price * (1 + adj) * escalation * float(RNG.normal(1.0, 0.012))
                if c.tier1_id == SURCHARGE_TIER1 and po_date >= SURCHARGE_FROM:
                    price *= 1 + SURCHARGE_PCT  # S3

                qty = int(max(25, RNG.normal(monthly_units / 5.5, monthly_units / 16.0)))

                lt_plan = int(RNG.integers(21, 63))
                delay = 0
                hit = disr[
                    (disr["country_code"] == country)
                    & (disr["start_date"] <= po_date)
                    & (disr["end_date"] >= po_date)
                ]
                if len(hit):
                    delay = int(hit["lead_time_impact_days"].max())
                lt_actual = int(max(7, lt_plan + delay + RNG.normal(1.5, 5.0)))

                ym = MONTH_STARTS[m].strftime("%Y-%m")
                fx = 1.0
                rows.append(
                    {
                        "po_id": f"PO-{po_no:06d}",
                        "po_date": po_date.strftime("%Y-%m-%d"),
                        "year_month": ym,
                        "part_id": c.part_id,
                        "tier1_id": c.tier1_id,
                        "contract_id": c.contract_id,
                        "plant_id": str(RNG.choice(["PL-ERL", "PL-AMB", "PL-CMG", "PL-LEN"])),
                        "order_qty": qty,
                        "document_currency": doc_ccy,
                        "fx_rate_used": fx,
                        "unit_price_eur": round(price, 4),
                        "net_value_eur": round(price * qty, 2),
                        "incoterm": c.incoterm,
                        "lead_time_planned_days": lt_plan,
                        "lead_time_actual_days": lt_actual,
                        "on_time_flag": bool(lt_actual <= lt_plan + 3),
                    }
                )
                po_no += 1

    df = pd.DataFrame(rows)
    # document-currency price for non-EUR suppliers (reporting stays EUR)
    fx = build_fx_lookup()
    df["fx_rate_used"] = [
        1.0 if c == "EUR" else fx[(c, ym)]
        for c, ym in zip(df["document_currency"], df["year_month"])
    ]
    df["unit_price_doc_ccy"] = (df["unit_price_eur"] / df["fx_rate_used"]).round(4)
    return df


_FX_CACHE: dict[tuple[str, str], float] = {}


def build_fx_lookup() -> dict[tuple[str, str], float]:
    return _FX_CACHE


# ---------------------------------------------------------------------------
# 9. map_tier1_tier2_declared   (S2 + S6)
# ---------------------------------------------------------------------------

S2_ENTITY = "E001"
# Three Tier-1s that genuinely consume aluminium (switchgear / busbar families)
# and that all trace back to the SAME Chinese smelter under different spellings.
S2_TIER1S = ["T1-004", "T1-005", "T1-008"]


def build_declared_map(
    df_raw2: pd.DataFrame, truth: pd.DataFrame, t1_metal_spend: pd.DataFrame
) -> pd.DataFrame:
    raw = df_raw2.merge(truth, on="raw_supplier_id", how="left")

    # ---- which (tier1, metal) combinations stay undeclared? -------------------
    combos = t1_metal_spend.sort_values(["tier1_id", "metal_id"]).reset_index(drop=True)
    total = float(combos["metal_spend_eur"].sum())
    protected = set(
        (t, "ALUMINIUM") for t in S2_TIER1S
    ) | {(combos.loc[combos["metal_spend_eur"].idxmax(), "tier1_id"],
          combos.loc[combos["metal_spend_eur"].idxmax(), "metal_id"])}

    order = RNG.permutation(len(combos))
    uncovered, undeclared = 0.0, set()
    for i in order:
        t, m, s = combos.loc[i, ["tier1_id", "metal_id", "metal_spend_eur"]]
        if (t, m) in protected:
            continue
        if (uncovered + s) / total <= TARGET_UNCOVERED_SHARE:
            undeclared.add((t, m))
            uncovered += float(s)

    # ---- build declarations ---------------------------------------------------
    rows = []
    did = 1
    for r in combos.itertuples(index=False):
        if (r.tier1_id, r.metal_id) in undeclared:
            continue

        if (r.tier1_id, r.metal_id) in [(t, "ALUMINIUM") for t in S2_TIER1S]:
            # S2 -- each Tier-1 names the SAME smelter, spelled differently
            variants = raw[raw["true_entity_id"] == S2_ENTITY].reset_index(drop=True)
            pick = variants.iloc[S2_TIER1S.index(r.tier1_id) % len(variants)]
            candidates = [pick]
            shares = [1.0]
        else:
            pool = raw[raw["supplied_metal_id"] == r.metal_id]
            if pool.empty:
                continue
            # one raw variant per distinct entity, to keep declarations plausible
            pool = pool.groupby("true_entity_id", as_index=False).first()
            n = int(min(len(pool), RNG.integers(1, 4)))
            picked = pool.sample(n=n, random_state=int(RNG.integers(0, 10**6)))
            candidates = [row for row in picked.itertuples(index=False)]
            w = RNG.dirichlet(np.ones(n) * 2.2)
            shares = [round(float(x), 4) for x in w]

        for cand, sh in zip(candidates, shares):
            rows.append(
                {
                    "declaration_id": f"DEC-{did:04d}",
                    "tier1_id": r.tier1_id,
                    "metal_id": r.metal_id,
                    "raw_supplier_id": cand.raw_supplier_id,
                    "declared_supplier_name": cand.raw_supplier_name,
                    "declared_country_code": cand.country_code,
                    "declared_spend_share": sh,
                    "declaration_date": (
                        START_DATE + pd.Timedelta(days=int(RNG.integers(180, 1000)))
                    ).strftime("%Y-%m-%d"),
                    "source_system": cand.source_system,
                    "self_declared_confidence": str(
                        RNG.choice(["High", "Medium", "Low"], p=[0.45, 0.4, 0.15])
                    ),
                    "verified_by_audit": bool(RNG.random() < 0.35),
                }
            )
            did += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 12. fact_supplier_compliance   (S4)
# ---------------------------------------------------------------------------

def build_compliance(
    df_raw2: pd.DataFrame, truth: pd.DataFrame, tier2_spend: dict[str, float]
) -> pd.DataFrame:
    raw = df_raw2.merge(truth, on="raw_supplier_id", how="left")
    # one compliance record per entity, keyed on ONE of its raw ids
    # (preferring the sustainability portal source -- as it would be in reality)
    pref = raw.sort_values(
        ["true_entity_id", "source_system"], key=lambda s: s.map(
            {"SUSTAINABILITY_PORTAL": 0, "SUPPLIER_SURVEY": 1, "ERP_SAP": 2}
        ) if s.name == "source_system" else s
    ).groupby("true_entity_id", as_index=False).first()

    top2 = sorted(tier2_spend.items(), key=lambda kv: -kv[1])[:2]
    top2_ids = {k for k, _ in top2}

    rows = []
    for r in pref.itertuples(index=False):
        risky = r.true_entity_id in top2_ids  # S4
        cyber = (
            round(float(RNG.uniform(28, 44)), 1) if risky
            else round(float(RNG.uniform(46, 93)), 1)
        )
        last_audit = AS_OF - pd.Timedelta(days=int(RNG.integers(620, 900)) if risky
                                          else int(RNG.integers(40, 700)))
        rows.append(
            {
                "compliance_id": f"CMP-T2-{r.true_entity_id[1:]}",
                "supplier_tier": "Tier-2",
                "supplier_ref_id": r.raw_supplier_id,
                "supplier_ref_name": r.raw_supplier_name,
                "country_code": r.country_code,
                "assessment_date": (AS_OF - pd.Timedelta(days=int(RNG.integers(30, 400)))
                                    ).strftime("%Y-%m-%d"),
                "cyber_maturity_score": cyber,
                "iso27001_flag": bool(cyber > 62),
                "tisax_flag": bool(cyber > 74),
                "nis2_in_scope": bool(RNG.random() < 0.55),
                "secure_data_exchange_flag": bool(cyber > 55),
                "cyber_incidents_24m": int(RNG.integers(2, 6) if risky else RNG.integers(0, 3)),
                "co2e_kg_per_kg_metal": round(float(RNG.uniform(1.8, 16.5)), 2),
                "renewable_energy_pct": round(float(RNG.uniform(4, 88)), 1),
                "csrd_reporting_flag": bool(RNG.random() < 0.5),
                "cbam_relevant": r.supplied_metal_id in ("ALUMINIUM", "STEEL_HRC"),
                "lksg_risk_class": str(RNG.choice(["Low", "Medium", "High"], p=[0.4, 0.4, 0.2])),
                "conflict_minerals_declared": bool(RNG.random() < 0.7),
                "last_audit_date": last_audit.strftime("%Y-%m-%d"),
                "audit_overdue_flag": bool((AS_OF - last_audit).days > 548),
            }
        )

    for sid, name, cc, _city, _f, _n, _s in TIER1:
        cyber = round(float(RNG.uniform(58, 96)), 1)
        last_audit = AS_OF - pd.Timedelta(days=int(RNG.integers(30, 600)))
        rows.append(
            {
                "compliance_id": f"CMP-{sid}",
                "supplier_tier": "Tier-1",
                "supplier_ref_id": sid,
                "supplier_ref_name": name,
                "country_code": cc,
                "assessment_date": (AS_OF - pd.Timedelta(days=int(RNG.integers(30, 300)))
                                    ).strftime("%Y-%m-%d"),
                "cyber_maturity_score": cyber,
                "iso27001_flag": bool(cyber > 62),
                "tisax_flag": bool(cyber > 74),
                "nis2_in_scope": True,
                "secure_data_exchange_flag": bool(cyber > 55),
                "cyber_incidents_24m": int(RNG.integers(0, 3)),
                "co2e_kg_per_kg_metal": round(float(RNG.uniform(1.2, 9.0)), 2),
                "renewable_energy_pct": round(float(RNG.uniform(15, 95)), 1),
                "csrd_reporting_flag": True,
                "cbam_relevant": False,
                "lksg_risk_class": str(RNG.choice(["Low", "Medium", "High"], p=[0.5, 0.4, 0.1])),
                "conflict_minerals_declared": True,
                "last_audit_date": last_audit.strftime("%Y-%m-%d"),
                "audit_overdue_flag": bool((AS_OF - last_audit).days > 548),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_date = build_dim_date()
    df_metal = build_dim_metal()
    df_index = build_fact_market_index()
    df_fx = build_fact_fx_rate()

    _FX_CACHE.update(
        {(r.currency_code, r.year_month): r.eur_per_unit
         for r in df_fx.itertuples(index=False)}
    )

    # EUR per kg per metal per month -- the common denominator of the whole model
    idx_eur_kg: dict[str, np.ndarray] = {}
    for mid, *_ in METALS:
        sub = df_index[df_index["metal_id"] == mid].sort_values("month_index")
        ccy = sub["quote_currency"].iloc[0]
        rates = np.array([_FX_CACHE[(ccy, ym)] for ym in sub["year_month"]])
        idx_eur_kg[mid] = (sub["settlement_price"].to_numpy() * rates) / 1000.0

    df_t1 = build_dim_supplier_tier1()
    df_raw2, df_truth = build_tier2_raw()
    df_country = build_dim_country_risk()
    df_disr = build_fact_disruption_events()
    df_part, df_bom = build_parts_and_bom(idx_eur_kg)

    # metal cost share per part x metal (at base index) -- drives contracts + pricing
    bom = df_bom.merge(df_part[["part_id", "standard_cost_eur"]], on="part_id")
    bom["base_metal_cost_eur"] = [
        round(r.effective_kg_per_unit * idx_eur_kg[r.metal_id][0], 4)
        for r in bom.itertuples(index=False)
    ]
    bom["cost_share"] = (bom["base_metal_cost_eur"] / bom["standard_cost_eur"]).round(6)
    bom_cost_share = bom[["part_id", "metal_id", "base_metal_cost_eur", "cost_share"]]
    df_bom = df_bom.merge(bom_cost_share, on=["part_id", "metal_id"], how="left")

    df_contract = build_contracts(df_part, bom_cost_share)
    df_po = build_purchase_orders(df_contract, df_part, bom_cost_share, idx_eur_kg, df_disr)

    # Tier-1 x metal spend exposure (needed before the n-tier declarations)
    spend = df_po.merge(df_bom[["part_id", "metal_id", "cost_share"]], on="part_id")
    spend["metal_spend_eur"] = spend["net_value_eur"] * spend["cost_share"]
    t1_metal_spend = (
        spend.groupby(["tier1_id", "metal_id"], as_index=False)["metal_spend_eur"]
        .sum()
        .round(2)
    )

    df_map = build_declared_map(df_raw2, df_truth, t1_metal_spend)

    # allocated Tier-2 spend -> identifies the two biggest sub-suppliers (S4)
    alloc = df_map.merge(t1_metal_spend, on=["tier1_id", "metal_id"], how="left")
    alloc = alloc.merge(df_truth, on="raw_supplier_id", how="left")
    alloc["allocated_eur"] = alloc["metal_spend_eur"] * alloc["declared_spend_share"]
    tier2_spend = alloc.groupby("true_entity_id")["allocated_eur"].sum().to_dict()

    df_comp = build_compliance(df_raw2, df_truth, tier2_spend)

    # -----------------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------------
    outputs = {
        "dim_date": df_date,
        "dim_metal": df_metal,
        "fact_market_index": df_index,
        "fact_fx_rate": df_fx,
        "dim_part": df_part,
        "bom_metal_content": df_bom,
        "dim_supplier_tier1": df_t1,
        "dim_supplier_tier2_raw": df_raw2,
        "map_tier1_tier2_declared": df_map,
        "dim_contract": df_contract,
        "fact_purchase_orders": df_po,
        "fact_supplier_compliance": df_comp,
        "dim_country_risk": df_country,
        "fact_disruption_events": df_disr,
        "_ground_truth_tier2": df_truth,
    }
    for name, df in outputs.items():
        df.to_csv(OUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    validate(outputs, idx_eur_kg, t1_metal_spend, alloc, tier2_spend)


# ---------------------------------------------------------------------------
# Validation + scenario report
# ---------------------------------------------------------------------------

def validate(outputs, idx_eur_kg, t1_metal_spend, alloc, tier2_spend) -> None:
    d = outputs
    po, part, bom, contract = d["fact_purchase_orders"], d["dim_part"], d["bom_metal_content"], d["dim_contract"]
    raw2, truth, mp = d["dim_supplier_tier2_raw"], d["_ground_truth_tier2"], d["map_tier1_tier2_declared"]

    # ---- referential integrity ------------------------------------------------
    assert set(po["part_id"]) <= set(part["part_id"]), "orphan part_id in PO"
    assert set(po["contract_id"]) <= set(contract["contract_id"]), "orphan contract_id"
    assert set(po["tier1_id"]) <= set(d["dim_supplier_tier1"]["tier1_id"]), "orphan tier1_id"
    assert set(bom["part_id"]) <= set(part["part_id"]), "orphan part_id in BOM"
    assert set(bom["metal_id"]) <= set(d["dim_metal"]["metal_id"]), "orphan metal_id in BOM"
    assert set(contract["index_metal_id"]) <= set(d["dim_metal"]["metal_id"]), "orphan index metal"
    assert set(mp["raw_supplier_id"]) <= set(raw2["raw_supplier_id"]), "orphan raw_supplier_id"
    assert set(raw2["country_code"]) <= set(d["dim_country_risk"]["country_code"]), "missing country"
    assert set(d["dim_supplier_tier1"]["country_code"]) <= set(
        d["dim_country_risk"]["country_code"]), "missing tier1 country"
    assert "true_entity_id" not in raw2.columns, "ground truth leaked into raw table"

    # ---- domain sanity --------------------------------------------------------
    assert (po["unit_price_eur"] > 0).all(), "non-positive price"
    assert (po["order_qty"] > 0).all(), "non-positive qty"
    assert (d["fact_market_index"]["settlement_price"] > 0).all(), "non-positive index"
    assert d["fact_market_index"]["month_index"].nunique() == N_MONTHS, "month count"
    assert po["po_id"].is_unique and contract["contract_id"].is_unique, "duplicate keys"
    assert 9_000 <= len(po) <= 14_000, f"PO row count out of range: {len(po):,}"
    assert len(part) == 50 and len(truth["true_entity_id"].unique()) == 20, "entity counts"

    # ---- scenario checks ------------------------------------------------------
    # S2: the hidden smelter is declared by >= 3 Tier-1s under >= 3 spellings
    s2 = mp.merge(truth, on="raw_supplier_id")
    s2 = s2[s2["true_entity_id"] == S2_ENTITY]
    assert s2["tier1_id"].nunique() >= 3, "S2: not enough Tier-1s behind the smelter"
    assert s2["declared_supplier_name"].nunique() >= 3, "S2: spellings not distinct"

    # S6: transparency coverage
    covered = mp[["tier1_id", "metal_id"]].drop_duplicates()
    cov = t1_metal_spend.merge(covered, on=["tier1_id", "metal_id"], how="inner")
    coverage = cov["metal_spend_eur"].sum() / t1_metal_spend["metal_spend_eur"].sum()
    assert 0.62 <= coverage <= 0.82, f"S6: coverage {coverage:.1%} off target"

    # S3: claimed vs. BOM-justified surcharge for the Tier-1 under negotiation
    t4 = po[po["tier1_id"] == SURCHARGE_TIER1]
    t4b = bom[bom["part_id"].isin(t4["part_id"].unique())]
    pre = t4[t4["po_date"] < SURCHARGE_FROM.strftime("%Y-%m-%d")]
    m_pre, m_post = _m(SURCHARGE_FROM) - 3, _m(SURCHARGE_FROM) + 3
    justified = 0.0
    weights = t4b.groupby("metal_id")["base_metal_cost_eur"].sum()
    shares = t4b.groupby("metal_id")["cost_share"].mean()
    for metal in weights.index:
        delta = idx_eur_kg[metal][m_post] / idx_eur_kg[metal][m_pre] - 1.0
        justified += float(shares[metal]) * delta
    gap_pp = (SURCHARGE_PCT - justified) * 100

    # S1: copper-led fixed-price spend expiring within 12 months of the as-of date
    cu_parts = set(contract.loc[contract["index_metal_id"] == "COPPER", "part_id"])
    fixed = contract[(contract["metal_clause_type"] == "FIXED_PRICE")
                     & (contract["index_metal_id"] == "COPPER")]
    cu_spend = po[po["part_id"].isin(cu_parts)]["net_value_eur"].sum()
    fixed_spend = po[po["contract_id"].isin(fixed["contract_id"])]["net_value_eur"].sum()
    s1_share = fixed_spend / cu_spend if cu_spend else 0.0

    # S4
    top2 = sorted(tier2_spend.items(), key=lambda kv: -kv[1])[:2]
    t2c = d["fact_supplier_compliance"]
    t2c = t2c[t2c["supplier_tier"] == "Tier-2"].merge(truth, left_on="supplier_ref_id",
                                                      right_on="raw_supplier_id", how="left")
    worst = t2c[t2c["true_entity_id"].isin([k for k, _ in top2])]
    assert (worst["cyber_maturity_score"] < 45).all(), "S4: top spend suppliers not flagged"

    # S5
    cu_usd = idx_eur_kg["COPPER"]  # already EUR; compare with pure-USD move
    idx = d["fact_market_index"]
    cu = idx[idx["metal_id"] == "COPPER"].sort_values("month_index")["settlement_price"].to_numpy()
    usd_move = cu[22] / cu[12] - 1
    eur_move = cu_usd[22] / cu_usd[12] - 1

    # ---- report ---------------------------------------------------------------
    line = "-" * 74
    print(line)
    print("PHASE 1 COMPLETE -- synthetic dataset written to")
    print(f"  {OUT_DIR}")
    print(line)
    print(f"{'table':<32}{'rows':>10}{'cols':>8}")
    print(line)
    for name, df in d.items():
        print(f"{name:<32}{len(df):>10,}{len(df.columns):>8}")
    print(line)
    print("REFERENTIAL INTEGRITY ........ OK (all foreign keys resolve)")
    print(line)
    print("SCENARIO CHECKS")
    print(f"  S1  copper spend on FIXED_PRICE clauses .......... {s1_share:>7.1%}")
    print(f"      fixed-price contracts expiring <=12m ......... {len(fixed):>7,}")
    print(f"  S2  Tier-1s pointing at 1 hidden smelter ......... {s2['tier1_id'].nunique():>7}"
          f"  ({s2['declared_supplier_name'].nunique()} spellings)")
    print(f"  S3  claimed surcharge (T1-004) ................... {SURCHARGE_PCT:>7.1%}")
    print(f"      BOM x index justified ........................ {justified:>7.1%}")
    print(f"      negotiation gap .............................. {gap_pp:>7.1f} pp")
    print(f"  S4  weakest cyber score among top-2 Tier-2s ...... "
          f"{worst['cyber_maturity_score'].min():>7.1f}")
    print(f"  S5  copper move M12->M22 in USD .................. {usd_move:>7.1%}")
    print(f"      same move in EUR (FX amplified) .............. {eur_move:>7.1%}")
    print(f"  S6  n-tier transparency coverage (by spend) ...... {coverage:>7.1%}")
    print(f"  S7  capped/lagged contracts ...................... "
          f"{(contract['metal_clause_type'] == 'CAPPED_INDEX').sum():>7,}")
    print(line)
    print(f"  Total purchasing spend ........... EUR {po['net_value_eur'].sum():>18,.0f}")
    print(f"  Embedded metal value ............. EUR "
          f"{t1_metal_spend['metal_spend_eur'].sum():>18,.0f}")
    print(f"  Observation window ............... {po['po_date'].min()} .. {po['po_date'].max()}")
    print(line)

    assert gap_pp >= 5.0, f"S3: negotiation gap too small ({gap_pp:.1f} pp)"
    assert s1_share >= 0.25, f"S1: fixed-price copper share too small ({s1_share:.1%})"
    assert eur_move > usd_move, "S5: FX did not amplify the copper move"
    print("ALL ASSERTIONS PASSED.\n")


if __name__ == "__main__":
    main()
