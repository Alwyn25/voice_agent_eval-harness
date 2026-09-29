"""
Build a rich knowledge document from:
1. DSR Highland Green brochure PDF  (project overview, specs, amenities)
2. Golden dataset Excel sheets       (pricing, charges, intents, tools)

Saves the combined text to eval_harness/knowledge_base.txt
which is what both the offline eval and the live backend ingest.
"""
import textwrap
from pathlib import Path

import pandas as pd

GOLDEN_XLSX  = r"C:\Users\ADVORA-ALWYN\OneDrive\Desktop\highland_greenz_golden_dataset.xlsx"
BROCHURE_TXT = Path(__file__).parent / "brochure_text.txt"
OUT          = Path(__file__).parent / "knowledge_base.txt"


def load_brochure() -> str:
    return BROCHURE_TXT.read_text(encoding="utf-8")


def load_price_matrix(xl) -> str:
    df = pd.read_excel(xl, sheet_name="Price_Matrix", header=None)
    lines = ["## DSR Highland Greenz — Price Matrix\n"]
    lines.append("Rate per sft: Rs.11,499 (builder confirmed for 3BHK 1239/1390/1560 sft)\n")
    lines.append("All other configurations are DERIVED at the same rate — confirm with sales.\n")
    lines.append("")
    # Parse the table from row 7+ (after assumptions)
    for i, row in df.iterrows():
        vals = [str(v).strip() for v in row if str(v).strip() and str(v) != "nan"]
        if vals:
            lines.append(" | ".join(vals))
    return "\n".join(lines)


def load_charges(xl) -> str:
    df = pd.read_excel(xl, sheet_name="Charges_Rules")
    df = df.fillna("")
    lines = ["## DSR Highland Greenz — Additional Charges\n"]
    for _, r in df.iterrows():
        charge  = str(r.get("Charge", "")).strip()
        amount  = str(r.get("Amount", "")).strip()
        cond    = str(r.get("Condition / trigger", "")).strip()
        source  = str(r.get("Source", "")).strip()
        if charge:
            lines.append(f"### {charge}")
            lines.append(f"Amount: {amount}")
            if cond:
                lines.append(f"Condition: {cond}")
            if source:
                lines.append(f"Source: {source}")
            lines.append("")
    return "\n".join(lines)


def load_intents(xl) -> str:
    df = pd.read_excel(xl, sheet_name="Intents")
    df = df.fillna("")
    lines = ["## Supported Question Types (Agent Intents)\n"]
    for _, r in df.iterrows():
        iid   = str(r.get("ID", "")).strip()
        intent = str(r.get("Intent", "")).strip()
        ex    = str(r.get("Example (English)", "")).strip()
        if iid and intent:
            lines.append(f"{iid}: {intent}")
            if ex:
                lines.append(f"  Example: {ex}")
    return "\n".join(lines)


def load_knowledge_gaps(xl) -> str:
    df = pd.read_excel(xl, sheet_name="Knowledge_Gaps")
    df = df.fillna("")
    lines = ["## Known Knowledge Gaps — Escalate These\n"]
    for _, r in df.iterrows():
        q   = str(r.get("Question the agent will be asked", "")).strip()
        st  = str(r.get("Status", "")).strip()
        beh = str(r.get("Required behaviour", "")).strip()
        if q:
            lines.append(f"Q: {q}")
            lines.append(f"Status: {st}")
            lines.append(f"Required behaviour: {beh}")
            lines.append("")
    return "\n".join(lines)


def load_project_facts() -> str:
    """Hard-coded key facts that should always be in the knowledge base."""
    return textwrap.dedent("""
    ## DSR Highland Greenz — Key Project Facts

    **Location**: Chikkanayakanahalli, off Sarjapur Road, Bengaluru
    **Total area**: 10.10 acres
    **Towers**: 4 towers across 8 wings (Tower A, B, C, D)
    **Total flats**: 900 units
    **Structure**: 1 basement + Ground + 12 floors (B/G+12)
    **Clubhouses**: 3 clubhouses totalling 23,156 sft

    ### Configuration Areas
    - 1 BHK: 619 – 780 sft (indicative, Rs.11,499/sft)
    - 1.5 BHK: derived (Rs.11,499/sft, sales to confirm)
    - 2 BHK: 875 – 1,210 sft (indicative, Rs.11,499/sft)
    - 3 BHK: 1,239 / 1,390 / 1,422 / 1,560 sft

    ### Builder-Confirmed Pricing (3BHK only)
    Rate per sft: Rs.11,499
    - 3BHK 1,239 sft: all-in approx Rs.1.54 crore (B/B parking)
    - 3BHK 1,390 sft: all-in approx Rs.1.72 crore (B/B parking)
    - 3BHK 1,560 sft (1st floor only, extended balcony): confirm with builder
    Infrastructure charge: Rs.250/sft (excl. GST)
    Clubhouse charge: Rs.3,25,000 (flat, excl. GST)
    Car parking – B/B (back-to-back): Rs.5,50,000
    Car parking – 2 singles: Rs.6,00,000

    ### Floor Rise
    Rs.30 per sft per floor from 4th floor onwards.

    ### Corner Flat Premium
    - Corner flat: Rs.50/sft
    - Corner & premium: Rs.100/sft

    ### Solar Water Heater (top floor only)
    Rs.95,000 (excl. GST)

    ### Common Area Maintenance (CAM)
    Rs.3.50/sft/month, payable upfront for 24 months (excl. GST)

    ### GST
    NOT in source — escalate to sales. Never compute or estimate.

    ### Registration & Stamp Duty
    At actuals — exact amounts NOT available. Escalate.

    ### Possession / Handover Date
    NOT in source — escalate. Never hedge a date.

    ### Nearby Landmarks
    - Wipro Corp: 2.5 km
    - RGA Tech Park: 2.2 km
    - Decathlon: 2.5 km
    - Manipal Hospital: 5.5 km
    - Ecospace: 6.7 km
    - Electronic City: 9 km
    - Marathahalli: 9 km

    ### Amenities (partial list)
    Main pool, Kids pool, Tennis court, Basketball court, Cricket practice nets,
    Outdoor gym, Jogging track, Cycle lane, Children's play area, Dog park,
    Rock climbing wall, Trampoline park, Barbeque zone, Amphitheatre, Skating rink,
    Youth hangout zone, Party lawn with stage, Seasonal stream, Yoga/meditation deck.
    """).strip()


def main():
    xl = pd.ExcelFile(GOLDEN_XLSX)

    sections = [
        "# DSR Highland Greenz — Complete Knowledge Base\n",
        load_project_facts(),
        "\n\n" + load_price_matrix(xl),
        "\n\n" + load_charges(xl),
        "\n\n" + load_knowledge_gaps(xl),
        "\n\n## Brochure Content\n\n" + load_brochure(),
    ]

    full = "\n".join(sections)
    OUT.write_text(full, encoding="utf-8")
    print(f"Knowledge base: {len(full):,} chars → {OUT}")
    print(f"Sections: brochure + price matrix + charges + gaps + project facts")


if __name__ == "__main__":
    main()
