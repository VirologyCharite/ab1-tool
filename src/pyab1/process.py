import sys
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import numpy as np
import matplotlib.pyplot as plt
import re
import argparse
from typing import Iterable

BASE_COLORS = {
    "G": "orange",
    "A": "red",
    "T": "green",
    "C": "blue",
}

# Here's the order according to
# https://www.reddit.com/r/bioinformatics/comments/buswph/manual_creation_of_chromatogram_from_ab1_sanger/
BASE_CHANNELS = {
    "G": "DATA9",
    "A": "DATA10",
    "T": "DATA11",
    "C": "DATA12",
}

# Note that the order here is critical (this is checked below).
BASES = list(BASE_CHANNELS)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pattern",
        required=True,
        help="The nucleotide pattern to search for.",
    )

    parser.add_argument(
        "--ab1",
        required=True,
        help="The ab1 file to read.",
    )

    parser.add_argument(
        "--plot-chromatogram",
        "--pc",
        action="store_true",
        help="Plot a chromatogram.",
    )

    parser.add_argument(
        "--plot-qualities",
        "--pq",
        action="store_true",
        help="Plot the qualities.",
    )

    parser.add_argument(
        "--include-pattern",
        action="store_true",
        help="Also show intensity information for the pattern.",
    )

    parser.add_argument(
        "--strict",
        action="store_false",
        dest="allow_ambiguous",
        help="Do not allow ambiguous nucleotide matches.",
    )

    parser.add_argument(
        "--window",
        default=0,
        type=int,
        help="The intensity data window around the peak to consider.",
    )

    return parser.parse_args()


def find_sequence_positions(sequence: str, search_pattern: str, allow_ambiguous: bool):
    """
    Find all positions where a search pattern occurs in a sequence.

    Args:
        sequence: The uppercase DNA sequence to search.
        search_pattern: The uppercase pattern to search for (supports IUPAC codes).
        allow_ambiguous: Whether to allow ambiguous nucleotides in search.

    Returns:
        list: List of starting positions (0-indexed) where pattern is found.
    """
    positions = []

    if allow_ambiguous:
        # Convert IUPAC codes to regex pattern
        iupac_dict = {
            "R": "[AG]",
            "Y": "[CT]",
            "S": "[GC]",
            "W": "[AT]",
            "K": "[GT]",
            "M": "[AC]",
            "B": "[CGT]",
            "D": "[AGT]",
            "H": "[ACT]",
            "V": "[ACG]",
            "N": "[ACGT]",
        }

        pattern = search_pattern
        for code, regex in iupac_dict.items():
            pattern = pattern.replace(code, regex)

        for match in re.finditer(pattern, sequence.upper()):
            positions.append(match.start())
    else:
        sequence_str = sequence.upper()

        for i in range(len(sequence_str) - len(search_pattern) + 1):
            if sequence_str[i : i + len(search_pattern)] == search_pattern:
                positions.append(i)

    return positions


def get_trace_data_at_position(
    record: SeqRecord,
    trimmed_seq: str,
    position: int,
    left_trim_count: int,
    peak_locations,
    trace_data,
    window: int = 0,
):
    """
    Get trace data at a specific sequence position

    Args:
        record: BioPython SeqRecord from ABI file.
        position: Sequence position (0-indexed).
        window: Window around position to analyze.

    Returns:
        dict: Trace intensities and frequencies
    """
    untrimmed_position = position + left_trim_count
    called_base = str(record.seq)[untrimmed_position]

    if (
        hasattr(record, "letter_annotations")
        and "phred_quality" in record.letter_annotations
    ):
        quality_score = record.letter_annotations["phred_quality"][untrimmed_position]
    else:
        quality_score = None

    intensities = {}
    called_base = trimmed_seq[position]
    assert str(record.seq)[untrimmed_position] == called_base
    peak_location = peak_locations[position]

    for base in BASES:
        # Get intensity around peak location
        start_idx = max(0, peak_location - window)
        end_idx = min(len(trace_data[base]), peak_location + window + 1)
        trace_window = trace_data[base][start_idx:end_idx]
        intensities[base] = max(trace_window)

    total_intensity = sum(intensities.values())
    frequencies = {}

    assert total_intensity > 0
    for base in BASES:
        frequencies[base] = intensities.get(base, 0) / total_intensity

    return {
        "position": position,
        "called_base": called_base,
        "quality_score": quality_score,
        "trace_intensities": intensities,
        "frequencies": frequencies,
        "total_intensity": total_intensity,
    }


def analyze_abi_file(
    filename: str,
    search_pattern: str,
    window: int = 0,
    allow_ambiguous: bool = False,
    include_pattern: bool = False,
) -> dict:
    """
    Complete analysis workflow using BioPython

    Args:
        filename: Path to ABI file.
        search_pattern: The uppercase DNA sequence pattern to search for.
        allow_ambiguous: Whether to allow IUPAC ambiguous codes.

    Returns:
        dict: Complete analysis results
    """
    print(f"Reading ABI file: {filename!r}")
    record = SeqIO.read(filename, "abi")
    record.seq = record.seq.upper()
    raw = record.annotations["abif_raw"]

    # Check that the nucleotide order in the file channels is as we have it in BASES.
    assert raw["FWO_1"] == b"".join(bytes(b, "ASCII") for b in BASES)

    trimmed_seq = record.annotations["abif_raw"]["PBAS1"].decode("utf-8").upper()
    left_trim_count = record.seq.find(trimmed_seq)
    right_trim_count = len(record.seq) - len(trimmed_seq) - left_trim_count

    if left_trim_count == -1:
        sys.exit(
            f"Could not find the PBAS1 sequence from {filename!r} in the Bio SeqRecord!"
        )

    print(f"Sequence ID: {record.id}")
    print(f"Sequence length: {len(record.seq)}")
    print(f"Trimmed sequence length: {len(trimmed_seq)}")
    print(f"Left trimmed count: {left_trim_count}")
    print(f"First 50 trimmed bases: {trimmed_seq[:50]}")

    avg_quality = np.mean(record.letter_annotations["phred_quality"])
    print(f"Average quality score: {avg_quality:.2f}")

    # Look for the pattern in the trimmed sequence.
    positions = find_sequence_positions(trimmed_seq, search_pattern, allow_ambiguous)

    if positions:
        print(f"\nPattern {search_pattern!r} found at (trimmed) positions: {positions}")
    else:
        extra = (
            ""
            if allow_ambiguous
            else " Maybe you shouldn't use --strict, so as to allow ambiguous matching?"
        )
        sys.exit(
            f"Pattern {search_pattern!r} could not be found in the trimmed "
            f"sequence.{extra}"
        )

    peak_locations = raw["PLOC1"]
    trace_data = dict((base, raw[channel]) for base, channel in BASE_CHANNELS.items())

    results = []
    for match_pos in positions:
        print(f"\nMatch position {match_pos}:")
        if include_pattern:
            # Look at the whole of the pattern match, and one position more.
            positions_to_analyze = range(match_pos, match_pos + len(search_pattern) + 5)
        else:
            # Just look at one base beyond the end of the pattern match.
            positions_to_analyze = [match_pos + len(search_pattern)]

        for pos in positions_to_analyze:
            print(f"  Analyzing trimmed position {pos}:")
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

            print(f"    Called base: {analysis['called_base']}")
            print(f"    Quality score: {analysis['quality_score']}")
            print(f"    Total intensity: {analysis['total_intensity']}")
            print(f"    Trace intensities: {analysis['trace_intensities']}")
            freqs = dict((k, f"{v:.3f}") for k, v in analysis["frequencies"].items())
            print(f"    Frequencies: {freqs}")

    return {
        "analyses": results,
        "left-trim-count": left_trim_count,
        "peak-locations": peak_locations,
        "positions": positions,
        "record": record,
        "right-trim-count": right_trim_count,
        "trimmed-seq": trimmed_seq,
    }


def plot_quality_scores(
    record: SeqRecord,
    positions: Iterable[int],
    left_trim_count: int,
    right_trim_count: int,
):
    quality_scores = record.letter_annotations["phred_quality"]
    plt.figure(figsize=(12, 4))
    # plt.plot(quality_scores, alpha=0.7)
    plt.scatter(list(range(len(quality_scores))), quality_scores, alpha=0.7)

    for x in left_trim_count, len(record.seq) - right_trim_count:
        plt.axvline(
            x=x,
            color="black",
            linestyle="--",
            alpha=0.5,
        )

    for x in positions:
        plt.axvline(
            x=x + left_trim_count,
            color="red",
            linestyle="--",
            alpha=0.5,
        )

    plt.xlabel("Untrimmed position")
    plt.ylabel("Quality Score")
    plt.title("Quality Scores Along Sequence")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_chromatogram(
    record, trimmed_seq, peak_locations, start=0, end=None, highlight_positions=None
):
    """
    Plot chromatogram using BioPython data

    Args:
        record: BioPython SeqRecord
        start (int): Start position for plotting
        end (int): End position for plotting
        highlight_positions (list): Positions to highlight
    """
    if end is None:
        end = len(trimmed_seq)

    trace_data = {}

    for base, channel in BASE_CHANNELS.items():
        if channel in record.annotations["abif_raw"]:
            trace_data[base] = record.annotations["abif_raw"][channel]

    plt.figure(figsize=(15, 6))

    if start < len(peak_locations) and end <= len(peak_locations):
        trace_start = peak_locations[start] if start < len(peak_locations) else 0
        trace_end = (
            peak_locations[min(end - 1, len(peak_locations) - 1)]
            if end <= len(peak_locations)
            else len(trace_data["A"])
        )

        x = range(trace_start, min(trace_end, len(trace_data["A"])))

        for base, color in BASE_COLORS.items():
            if base in trace_data:
                y = trace_data[base][trace_start:trace_end]
                plt.plot(x, y[: len(x)], color=color, label=base, alpha=0.7)

        if highlight_positions:
            for pos in highlight_positions:
                if start <= pos < end and pos < len(peak_locations):
                    peak_loc = peak_locations[pos]
                    plt.axvline(
                        x=peak_loc,
                        color="black",
                        linestyle="--",
                        alpha=0.5,
                        label=f"Pos {pos} ({trimmed_seq[pos]})"
                        if pos == highlight_positions[0]
                        else "",
                    )

        plt.xlabel("Trace Index")
        plt.ylabel("Signal Intensity")
        plt.title(f"Chromatogram (positions {start}-{end - 1})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("Peak location data not available for plotting")


def main():
    args = get_args()
    pattern = args.pattern.upper()
    results = analyze_abi_file(
        args.ab1,
        pattern,
        window=args.window,
        allow_ambiguous=args.allow_ambiguous,
        include_pattern=args.include_pattern,
    )

    if args.plot_qualities:
        plot_quality_scores(
            results["record"],
            results["positions"],
            results["left-trim-count"],
            results["right-trim-count"],
        )

    if args.plot_chromatogram and results["positions"]:
        trimmed_seq = results["trimmed-seq"]
        first_pos = results["positions"][0]
        last_pos = results["positions"][-1]

        plot_chromatogram(
            results["record"],
            trimmed_seq,
            results["peak-locations"],
            start=max(0, first_pos - len(pattern)),
            end=min(len(trimmed_seq), last_pos + len(pattern) + 20),
            highlight_positions=[p + len(pattern) for p in results["positions"]],
        )


if __name__ == "__main__":
    main()
