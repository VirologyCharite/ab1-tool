import sys
import argparse
from Bio import SeqIO
import polars as pl
from typing import Dict, Any

from pyab1.process import (
    BASE_CHANNELS,
    BASES,
    find_sequence_positions,
    get_trace_data_at_position,
)


def analyze_abi_file_silent(
    filename: str,
    search_pattern: str,
    window: int = 0,
    allow_ambiguous: bool = False,
) -> Dict[str, Any]:
    """
    Silent version of analyze_abi_file that returns data without printing.

    Args:
        filename (str): Path to ABI file
        search_pattern (str): DNA sequence pattern to search for
        window (int): Window around position to analyze
        allow_ambiguous (bool): Whether to allow IUPAC ambiguous codes

    Returns:
        dict: Analysis results including trace intensities for positions after
        pattern matches
    """
    record = SeqIO.read(filename, "abi")
    raw = record.annotations["abif_raw"]

    # Check that the nucleotide order in the file channels is as we have it in BASES.
    assert raw["FWO_1"] == b"".join(bytes(b, "ASCII") for b in BASES)

    trimmed_seq = record.annotations["abif_raw"]["PBAS1"].decode("utf-8")
    left_trim_count = str(record.seq).find(trimmed_seq)

    if left_trim_count == -1:
        raise ValueError(
            f"Could not find the PBAS1 sequence from {filename!r} in the Bio SeqRecord!"
        )

    # Look for the pattern in the trimmed sequence
    positions = find_sequence_positions(trimmed_seq, search_pattern, allow_ambiguous)

    if not positions:
        raise ValueError(
            f"Pattern {search_pattern!r} could not be found in the trimmed sequence "
            f"of {filename!r}"
        )

    peak_locations = raw["PLOC1"]
    trace_data = dict((base, raw[channel]) for base, channel in BASE_CHANNELS.items())

    results = []
    for match_pos in positions:
        # Look at one base beyond the end of the pattern match
        pos = match_pos + len(search_pattern)

        analysis = get_trace_data_at_position(
            record,
            trimmed_seq,
            pos,
            left_trim_count,
            peak_locations,
            trace_data,
            window=window,
        )
        results.append(analysis)

    return {
        "filename": filename,
        "analyses": results,
        "match_count": len(positions),
    }


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process multiple AB1 files and generate Excel summary of nucleotide "
            "intensities."
        )
    )

    parser.add_argument(
        "--pattern",
        required=True,
        help="The nucleotide pattern to search for.",
    )

    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="The output Excel file path (e.g., results.xlsx).",
    )

    parser.add_argument(
        "--window",
        default=0,
        type=int,
        help="The intensity data window around the peak to consider.",
    )

    parser.add_argument(
        "--strict",
        action="store_false",
        dest="allow_ambiguous",
        help="Do not allow ambiguous nucleotide matches.",
    )

    parser.add_argument(
        "ab1_files",
        nargs="+",
        help="One or more AB1 files to process.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    all_results = []

    for ab1_file in args.ab1_files:
        try:
            result = analyze_abi_file_silent(
                ab1_file,
                args.pattern,
                window=args.window,
                allow_ambiguous=args.allow_ambiguous,
            )

            if result["match_count"] > 1:
                print(
                    f"WARNING: Pattern '{args.pattern}' found {result['match_count']} "
                    f"times in {ab1_file}",
                    file=sys.stderr,
                )

            # Add each match as a separate row
            for analysis in result["analyses"]:
                row_data = {
                    "Filename": ab1_file,
                    "G": analysis["trace_intensities"]["G"],
                    "A": analysis["trace_intensities"]["A"],
                    "T": analysis["trace_intensities"]["T"],
                    "C": analysis["trace_intensities"]["C"],
                }
                all_results.append(row_data)

        except Exception as e:
            print(f"ERROR processing {ab1_file}: {e}", file=sys.stderr)
            continue

    if not all_results:
        print(
            "ERROR: No results to write. No files were successfully processed.",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pl.DataFrame(all_results)

    summary_row = pl.DataFrame(
        [
            {
                "Filename": "TOTAL",
                "G": df["G"].sum(),
                "A": df["A"].sum(),
                "T": df["T"].sum(),
                "C": df["C"].sum(),
            }
        ]
    )

    # Combine data with summary
    df_with_summary = pl.concat([df, summary_row])

    df_with_summary.write_excel(args.output)

    print(f"Processed {len(args.ab1_files)} file(s)")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
