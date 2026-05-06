#!/usr/bin/env python3
"""
Convert beer game scenario logs into one Excel-compatible XML workbook.

Output:
  Simulation/beer_logs/beer_scenarios_35weeks.xml

Open the XML file directly in Microsoft Excel.
"""

import os
import re
from xml.sax.saxutils import escape


SCENARIOS = [
    ("scenario1_max_inventory_min_backlog_35weeks.log", "Scenario1_MaxInv"),
    ("scenario2_average_inventory_average_backlog_35weeks.log", "Scenario2_Average"),
    ("scenario3_optimized_min_total_cost_35weeks.log", "Scenario3_Optimized"),
]

ROLE_ORDER = ["Retailer", "Wholesaler", "Distributor", "Factory"]

SUMMARY_RE = re.compile(
    r"^(Retailer|Wholesaler|Distributor|Factory)\s+\| "
    r"In:\s*(\d+)\s+Recv:\s*(\d+)\s+Ship:\s*(\d+)\s+"
    r"InvEnd:\s*(\d+)\s+BackEnd:\s*(\d+)\s+"
    r"InvCost:\s*([0-9]+(?:\.[0-9]+)?)\s+BackCost:\s*([0-9]+(?:\.[0-9]+)?)\s+"
    r"WeekCost:\s*([0-9]+(?:\.[0-9]+)?)\s+Total:\s*([0-9]+(?:\.[0-9]+)?)\s*$"
)

REC_RE_ROLE = re.compile(
    r"^(Retailer|Wholesaler|Distributor)\s*->.*:\s*(\d+)\s+\(target=(\d+)\s+weeks\)\s*$"
)
REC_RE_FACTORY = re.compile(
    r"^Factory production\s*:\s*(\d+)\s+\(target=(\d+)\s+weeks\)\s*$"
)
WEEK_SUMMARY_RE = re.compile(r"^Week\s+(\d+)\s+Summary$")
WEEK_REC_RE = re.compile(r"^Week\s+(\d+)\s+Recommended Orders")


def parse_log(path):
    rows = {}
    recs = {}
    current_summary_week = None
    current_rec_week = None

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            m = WEEK_SUMMARY_RE.match(line)
            if m:
                current_summary_week = int(m.group(1))
                continue

            m = WEEK_REC_RE.match(line)
            if m:
                current_rec_week = int(m.group(1))
                continue

            m = SUMMARY_RE.match(line)
            if m and current_summary_week is not None:
                role = m.group(1)
                rows[(current_summary_week, role)] = {
                    "Week": current_summary_week,
                    "Role": role,
                    "In": int(m.group(2)),
                    "Recv": int(m.group(3)),
                    "Ship": int(m.group(4)),
                    "InvEnd": int(m.group(5)),
                    "BackEnd": int(m.group(6)),
                    "InvCost": float(m.group(7)),
                    "BackCost": float(m.group(8)),
                    "WeekCost": float(m.group(9)),
                    "Total": float(m.group(10)),
                }
                continue

            m = REC_RE_ROLE.match(line)
            if m and current_rec_week is not None:
                role = m.group(1)
                recs[(current_rec_week, role)] = {
                    "RecommendedOrder": int(m.group(2)),
                    "TargetWeeks": int(m.group(3)),
                }
                continue

            m = REC_RE_FACTORY.match(line)
            if m and current_rec_week is not None:
                recs[(current_rec_week, "Factory")] = {
                    "RecommendedOrder": int(m.group(1)),
                    "TargetWeeks": int(m.group(2)),
                }

    out = []
    for week in sorted({k[0] for k in rows.keys()}):
        for role in ROLE_ORDER:
            key = (week, role)
            if key not in rows:
                continue
            row = rows[key].copy()
            row["RecommendedOrder"] = recs.get(key, {}).get("RecommendedOrder", "")
            row["TargetWeeks"] = recs.get(key, {}).get("TargetWeeks", "")
            out.append(row)
    return out


def cell(value, value_type="String"):
    if value == "":
        return '<Cell><Data ss:Type="String"></Data></Cell>'
    if value_type == "Number":
        return f'<Cell><Data ss:Type="Number">{value}</Data></Cell>'
    return f'<Cell><Data ss:Type="String">{escape(str(value))}</Data></Cell>'


def worksheet_xml(name, rows):
    headers = [
        "Week",
        "Role",
        "In",
        "Recv",
        "Ship",
        "InvEnd",
        "BackEnd",
        "InvCost",
        "BackCost",
        "WeekCost",
        "Total",
        "RecommendedOrder",
        "TargetWeeks",
    ]
    number_cols = {
        "Week",
        "In",
        "Recv",
        "Ship",
        "InvEnd",
        "BackEnd",
        "InvCost",
        "BackCost",
        "WeekCost",
        "Total",
        "RecommendedOrder",
        "TargetWeeks",
    }

    lines = [f'<Worksheet ss:Name="{escape(name)}">', "<Table>"]
    lines.append("<Row>" + "".join(cell(h, "String") for h in headers) + "</Row>")

    for row in rows:
        row_cells = []
        for h in headers:
            if h in number_cols and row[h] != "":
                row_cells.append(cell(row[h], "Number"))
            else:
                row_cells.append(cell(row[h], "String"))
        lines.append("<Row>" + "".join(row_cells) + "</Row>")

    lines.append("</Table>")
    lines.append("</Worksheet>")
    return "\n".join(lines)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    workbook_path = os.path.join(base_dir, "beer_scenarios_35weeks.xml")

    sheets = []
    for filename, sheet_name in SCENARIOS:
        path = os.path.join(base_dir, filename)
        rows = parse_log(path)
        sheets.append(worksheet_xml(sheet_name, rows))

    xml = [
        '<?xml version="1.0"?>',
        '<?mso-application progid="Excel.Sheet"?>',
        '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:x="urn:schemas-microsoft-com:office:excel" '
        'xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet" '
        'xmlns:html="http://www.w3.org/TR/REC-html40">',
        *sheets,
        "</Workbook>",
    ]

    with open(workbook_path, "w", encoding="utf-8") as f:
        f.write("\n".join(xml))

    print(f"Created Excel-compatible workbook: {workbook_path}")


if __name__ == "__main__":
    main()
