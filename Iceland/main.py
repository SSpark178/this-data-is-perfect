"""SIGMOID Iceland: repair legacy geothermal monitoring exports.

Reads test-input.csv and writes test-output.csv in the process working directory.
Only Python's standard library is required.
"""
import csv
from decimal import Decimal, ROUND_HALF_UP


def repair_name(name):
    """Reverse repeated UTF-8 bytes accidentally decoded as legacy text."""
    name = name.strip()
    for _ in range(5):
        if not any(marker in name for marker in ("Ã", "Â", "â", "Ð", "�")):
            break
        for encoding in ("latin-1", "cp1252"):
            try:
                repaired = name.encode(encoding).decode("utf-8")
            except (UnicodeError, ValueError):
                continue
            if repaired != name:
                name = repaired
                break
        else:
            break
    return name


def main():
    readings = {}
    with open("test-input.csv", encoding="utf-8-sig", newline="") as source:
        lines = (line for line in source if line.strip() and not line.lstrip().startswith("#"))
        for row in csv.DictReader(lines, delimiter=";"):
            reading_id = int(row["reading_id"].strip())
            name = repair_name(row["station_name"])
            temp = Decimal(row["temperature_raw"].strip().replace(",", "."))
            unit = row["unit"].strip().upper()
            if unit == "F":
                temp = (temp - Decimal(32)) * Decimal(5) / Decimal(9)
            elif unit != "C":
                raise ValueError(f"Unexpected unit: {unit}")
            temp = temp.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
            readings[reading_id] = [reading_id, name, f"{temp:.1f}", row["reading_date"].strip()]

    with open("test-output.csv", "w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(("reading_id", "station_name", "temperature_c", "reading_date"))
        for reading_id in sorted(readings):
            writer.writerow(readings[reading_id])


if __name__ == "__main__":
    main()
