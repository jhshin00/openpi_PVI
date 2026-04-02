#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib


CATEGORY_ORDER = [
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
]


def _load_summary(input_json: pathlib.Path) -> dict:
    with open(input_json, "r") as f:
        return json.load(f)


def _write_wide_csv(data: dict, output_csv: pathlib.Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["suite", "total_success_rate", "total_episodes", "total_successes", *CATEGORY_ORDER]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for suite in sorted(data):
            summary = data[suite]
            row = {
                "suite": suite,
                "total_success_rate": summary.get("total_success_rate", 0.0),
                "total_episodes": summary.get("total_episodes", 0),
                "total_successes": summary.get("total_successes", 0),
            }
            per_category = summary.get("per_category", {})
            for category in CATEGORY_ORDER:
                row[category] = per_category.get(category, {}).get("success_rate", "")
            writer.writerow(row)


def _write_long_csv(data: dict, output_csv: pathlib.Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["suite", "category", "episodes", "successes", "success_rate"]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for suite in sorted(data):
            per_category = data[suite].get("per_category", {})
            for category in CATEGORY_ORDER:
                stats = per_category.get(category, {})
                writer.writerow(
                    {
                        "suite": suite,
                        "category": category,
                        "episodes": stats.get("episodes", 0),
                        "successes": stats.get("successes", 0),
                        "success_rate": stats.get("success_rate", ""),
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LIBERO-plus aggregate summary JSON to CSV.")
    parser.add_argument("--input-json", required=True, help="Path to per_suite_category_summary.json")
    parser.add_argument("--output-csv", required=True, help="Path to wide table CSV output")
    parser.add_argument("--output-long-csv", required=True, help="Path to long-format CSV output")
    args = parser.parse_args()

    input_json = pathlib.Path(args.input_json)
    output_csv = pathlib.Path(args.output_csv)
    output_long_csv = pathlib.Path(args.output_long_csv)

    data = _load_summary(input_json)
    _write_wide_csv(data, output_csv)
    _write_long_csv(data, output_long_csv)

    print(f"Wrote wide CSV to {output_csv}")
    print(f"Wrote long CSV to {output_long_csv}")


if __name__ == "__main__":
    main()
