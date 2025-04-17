import argparse
from dataclasses import dataclass, field
from typing import List, Tuple
from math import radians, sin, cos, sqrt, atan2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from collections.abc import Iterator
from itertools import tee, islice
from tabulate import tabulate
import csv

@dataclass
class BaseSegment:
    start_distance: float
    end_distance: float
    haversine_distance: float
    tp_distance: float
    start_elevation: float
    end_elevation: float
    elevation_change: float
    grade: float


@dataclass
class CommonGradeSegment:
    base_segments: List[BaseSegment] = field(default_factory=list)
    start_distance: float = 0.0
    end_distance: float = 0.0
    start_elevation: float = 0.0
    end_elevation: float = 0.0
    total_distance: float = 0.0
    altitude_change: float = 0.0
    average_grade: float = 0.0
    max_grade: float = 0.0


def diagnostic_print_list(title, items, toFile=None):
    """
    Prints a list/iterator of items with a title.
    """
    print(f"\n{title}\n{'-' * len(title)}")

    if isinstance(items, Iterator):
        print('items is an iterator')
        printable_items, items = tee(items)
    else:
        printable_items = items

    [print(f"{i + 1}. {item}") for i, item in enumerate(printable_items)]

    return items


def diagnostic_make_plot(window_title, x, x_title, y1, y1_title, y1_color, *args):
    """
    Create a plot with one X-axis and multiple Y-axes.

    Args:
        window_title (str): Title for the plot window/tab.
        x (list): The X-axis data.
        x_title (str): Label for the X-axis.
        y1 (list): Data for the first Y-axis.
        y1_title (str): Title for the first Y-axis.
        y1_color (str): Color for the first Y-axis.
        *args: Alternating Y-axis data, title, and color for additional Y-axes 
               (e.g., y2, y2_title, y2_color, y3, y3_title, y3_color, ...).
    """
    if len(args) % 3 != 0:
        raise ValueError("Each additional Y-axis data series must have a corresponding title and color.")

    # Create the figure and the first Y-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.manager.set_window_title(window_title)

    # Plot the first Y-axis
    ax1.plot(x, y1, label=y1_title, color=y1_color)
    ax1.set_xlabel(x_title, color="black")
    ax1.set_ylabel(y1_title, color=y1_color)
    ax1.tick_params(axis='y', labelcolor=y1_color)
    ax1.grid(True)

    # Add additional Y-axes if needed
    axes = [ax1]
    for i in range(0, len(args), 3):
        y_data = args[i]
        y_title = args[i + 1]
        y_color = args[i + 2]
        ax = ax1.twinx()  # Create a new Y-axis
        ax.spines['right'].set_position(('outward', 60 * (len(axes) - 1)))  # Offset each additional Y-axis
        ax.plot(x, y_data, label=y_title, color=y_color)
        ax.set_ylabel(y_title, color=y_color)
        ax.tick_params(axis='y', labelcolor=y_color)
        axes.append(ax)

    # Add the title and legends
    fig.suptitle(window_title)
    for ax in axes:
        ax.legend(loc="upper left")


def pct_diff(x,y):
    """
    Calculate the percent difference between two numbers
    """
    return (x-y)/x * 100.0


# Example usage of itertools
def window(iterable, size):
    """
    Create a sliding window of the specified size over the iterable

    Args:
        iterable: The input iterable
        size: The size of the sliding window (size of 2 returns adjacent pairs)
    Returns:
        A zip object containing the sliding windows

    Example:
        my_list = [1, 2, 3, 4, 5]
        paired = list(window(my_list, 2))
        print(paired)
        [(1, 2), (2, 3), (3, 4), (4, 5)]
    """
    iterables = tee(iterable, size)
    for i, it in enumerate(iterables):
        next(islice(it, i, i), None)
    return zip(*iterables)


def haversine(lat1, lon1, lat2, lon2):
    """
    Haversine formula to calculate horizontal distance from two lat/lon geocoords
    """
    R = 6371000  # Earth's radius in meters
    phi1, phi2 = radians(lat1), radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)
    a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Parse the TCX file
def parse_tcx_file(file_path):
    """
    Use garming training center database to parse the tcx file

    Args:
        file_path of tcx file

    Returns:
        parsed out tcx trackpoint objects     
    """
    namespace = {"ns": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}
    tree = ET.parse(file_path)
    root = tree.getroot()
    trackpoints = root.findall(".//ns:Trackpoint", namespace)
    return trackpoints, namespace


def extract_data_from_trackpoints(tcx_trackpoints, namespace):
    """
    Extact the relevant raw data from the TCX trackpoint data

    Args:
        tcx_trackpoints
        namespace

    Returns:

    """
    return [ {
                'LatitudeDegrees'  : float(tp.find("ns:Position/ns:LatitudeDegrees", namespace).text),
                'LongitudeDegrees' : float(tp.find("ns:Position/ns:LongitudeDegrees", namespace).text),
                'DistanceMeters'   : float(tp.find("ns:DistanceMeters", namespace).text),
                'AltitudeMeters'   : float(tp.find("ns:AltitudeMeters", namespace).text),
             }
             for tp in tcx_trackpoints
            ]


def compute_trackpoint_deltas(tp1, tp2):
    """
    Compute deltas between two trackpoints
    Args:
        tp1: trackpoint 1
        tp2: trackpoint 2
    Returns:
        A tuple of deltas:
            segment_start_distance_meters,
            segment_end_distance_meters,
            segment_haversine_distance_meters,
            segment_distance_meters,
            segment_start_elevation_meters,
            segment_end_elevation_meters,
            segment_elevation_meters,
            segment_grade,
    """
    segment_start_distance_meters    = tp1['DistanceMeters']
    segment_end_distance_meters      = tp2['DistanceMeters']
    segment_haversine_distance_meters= haversine(
                                        tp1['LatitudeDegrees'], tp1['LongitudeDegrees'],
                                        tp2['LatitudeDegrees'], tp2['LongitudeDegrees']
                                        )
    segment_distance_meters         = tp2['DistanceMeters'] - tp1['DistanceMeters']
    segment_start_elevation_meters  = tp1['AltitudeMeters']
    segment_end_elevation_meters    = tp2['AltitudeMeters']
    segment_elevation_meters        = tp2['AltitudeMeters'] - tp1['AltitudeMeters']
    segment_grade                   = segment_elevation_meters / segment_distance_meters
    print(f"segment_grade: {segment_grade:.2f}")
    if segment_grade < -1.0:
        print(f"-- BAD -- segment_grade: {segment_grade:.2f}")
    return (
        segment_start_distance_meters,
        segment_end_distance_meters,
        segment_haversine_distance_meters,
        segment_distance_meters,
        segment_start_elevation_meters,
        segment_end_elevation_meters,
        segment_elevation_meters,
        segment_grade,
        )


def compute_base_segments(adjacent_trackpoints):
    """
    Between each two segments compute the delta values.

    Args:
        adjacent_trackpoints: Pairs of adjacent trackpoints.

    Returns:
        List of BaseSegment objects.
    """
    return [
        BaseSegment(
            start_distance=tp1['DistanceMeters'],
            end_distance=tp2['DistanceMeters'],
            haversine_distance=haversine(
                tp1['LatitudeDegrees'], tp1['LongitudeDegrees'],
                tp2['LatitudeDegrees'], tp2['LongitudeDegrees']
            ),
            tp_distance=tp2['DistanceMeters'] - tp1['DistanceMeters'],
            start_elevation=tp1['AltitudeMeters'],
            end_elevation=tp2['AltitudeMeters'],
            elevation_change=tp2['AltitudeMeters'] - tp1['AltitudeMeters'],
            grade=(tp2['AltitudeMeters'] - tp1['AltitudeMeters']) /
                  (tp2['DistanceMeters'] - tp1['DistanceMeters']) if tp2['DistanceMeters'] - tp1['DistanceMeters'] > 0 else 0.0
        )
        for tp1, tp2 in adjacent_trackpoints
    ]


def merge_segments_by_similar_grade(base_segments: List[BaseSegment], min_distance=30, grade_threshold=1.0, max_segment_length=100):
    """
    Merge base segments into common grade segments based on improved criteria.

    Args:
        base_segments: List of BaseSegment objects.
        min_distance: Minimum distance (in meters) for a segment to be considered significant.
        grade_threshold: Maximum grade difference to merge segments.
        max_segment_length: Maximum length of a common grade segment.

    Returns:
        List of CommonGradeSegment objects.
    """
    common_grade_segments = []
    current_segment = None

    for segment in base_segments:
        # If no current segment, start a new one
        if current_segment is None:
            current_segment = CommonGradeSegment(
                base_segments=[segment],
                start_distance=segment.start_distance,
                end_distance=segment.end_distance,
                start_elevation=segment.start_elevation,
                end_elevation=segment.end_elevation,
                total_distance=segment.haversine_distance,
                altitude_change=segment.elevation_change,
                average_grade=segment.grade,
                max_grade=segment.grade,
            )
            continue

        # Calculate the grade difference and check if the segment can be merged
        grade_diff = abs(current_segment.average_grade - segment.grade)
        if grade_diff <= grade_threshold and current_segment.total_distance + segment.haversine_distance <= max_segment_length:
            current_segment.base_segments.append(segment)
            current_segment.end_distance = segment.end_distance
            current_segment.end_elevation = segment.end_elevation
            current_segment.total_distance += segment.haversine_distance
            current_segment.altitude_change += segment.elevation_change
            current_segment.max_grade = max(current_segment.max_grade, segment.grade)
            current_segment.average_grade = (
                current_segment.altitude_change / current_segment.total_distance
            )
        else:
            # Finalize the current segment if it meets the minimum distance
            if current_segment.total_distance >= min_distance:
                common_grade_segments.append(current_segment)
            # Start a new segment
            current_segment = CommonGradeSegment(
                base_segments=[segment],
                start_distance=segment.start_distance,
                end_distance=segment.end_distance,
                start_elevation=segment.start_elevation,
                end_elevation=segment.end_elevation,
                total_distance=segment.haversine_distance,
                altitude_change=segment.elevation_change,
                average_grade=segment.grade,
                max_grade=segment.grade,
            )

    # Add the last segment if it meets the minimum distance
    if current_segment and current_segment.total_distance >= min_distance:
        common_grade_segments.append(current_segment)

    return common_grade_segments


def summarize_common_grade_segments(common_grade_segments: List[CommonGradeSegment]):
    """
    Summarize the common grade segments by sorting them based on the number of base segments.

    Args:
        common_grade_segments: List of CommonGradeSegment objects.
    """
    # Sort the segments by the number of base_segments in reverse order
    sorted_segments = sorted(
        common_grade_segments,
        key=lambda seg: len(seg.base_segments),
        reverse=True
    )

    # Calculate totals
    total_common_segments = len(common_grade_segments)
    total_base_segments = sum(len(seg.base_segments) for seg in common_grade_segments)

    # Print the summary
    print("\nSummary of Common Grade Segments (sorted by number of base segments):")
    print(f"Total Common Grade Segments: {total_common_segments}")
    print(f"Total Base Segments: {total_base_segments}")
    print(f"{'Index':<6}{'Start Distance (m)':<20}{'Avg Grade (%)':<15}{'Max Grade (%)':<15}{'# Base Segments':<15}")
    print("-" * 70)

    for idx, segment in enumerate(sorted_segments):
        print(
            f"{idx:<6}{segment.start_distance:<20.2f}{segment.average_grade:<15.2f}"
            f"{segment.max_grade:<15.2f}{len(segment.base_segments):<15}"
        )



def plot_common_grade_segments(common_grade_segments):
    """
    Add the common grade segments plot to the diagnostic plots window as a separate tab.

    Args:
        common_grade_segments: List of merged segments with aggregated data.
    """
    # Create a new figure for the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.canvas.manager.set_window_title("Common Grade Segments")

    colors = ["blue", "red"]  # Alternating colors

    for i, segment in enumerate(common_grade_segments):
        # Extract x (distance) and y (elevation) data for the segment
        x = [bs.start_distance for bs in segment.base_segments] + [ segment.end_distance ]
        y = [bs.start_elevation for bs in segment.base_segments] + [ segment.end_elevation ]

        # Plot the segment with alternating colors
        ax.plot(
            x,
            y,
            label=f"Segment {i + 1} (Grade: {segment.average_grade:.1f}%)",
            color=colors[i % len(colors)],
        )

        # Annotate the max grade point
        max_grade_idx = max(
            range(len(segment.base_segments)),
            key=lambda idx: segment.base_segments[idx].grade,
        )
        max_grade_point = segment.base_segments[max_grade_idx]
        ax.annotate(
            f"Max Grade: {max_grade_point.grade:.1f}%",
            (max_grade_point.start_distance, max_grade_point.start_elevation),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color="black",
            fontsize=8,
        )

    # Set labels, title, and legend
    ax.set_xlabel("Distance (meters)")
    ax.set_ylabel("Elevation (meters)")
    ax.set_title("Common Grade Segments")
    ax.legend()
    ax.grid(True)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Find steepest segments in a TCX file.")
    parser.add_argument("file_path", type=str, help="Path to the TCX file.")
    parser.add_argument("--min_distance", type=float, default=50, help="Minimum segment distance in meters (default: 50).")
    parser.add_argument("--grade_delta", type=float, default=5, help="Maximum grade difference to merge segments (default: 5).")
    parser.add_argument("--min_grade", type=float, default=7, help="Minimum steep grade to report on (default: 7%).")

    # Parse arguments
    args = parser.parse_args()

    tcx_trackpoints, namespace = parse_tcx_file(args.file_path)

    # Parse the TCX file
    trackpoints = extract_data_from_trackpoints(tcx_trackpoints, namespace)
    # trackpoints = diagnostic_print_list("Extracted Trackpoints", trackpoints)

    # diagnostic_make_plot(
    #     "Trackpoint Elevation Profile",
    #     [tp['DistanceMeters'] for tp in trackpoints], "Distance (meters)",
    #     [tp['AltitudeMeters'] for tp in trackpoints], "Elevation (meters)", "blue"
    # )

    adjacent_trackpoints = window(trackpoints, 2)
    # adjacent_trackpoints = diagnostic_print_list("Adjacent Trackpoints", adjacent_trackpoints)

    base_segments = compute_base_segments(adjacent_trackpoints)

    base_segments = diagnostic_print_list("Base Segments", base_segments, toFile='base_segments.txt')

    # diagnostic_make_plot(
    #     "Adjacent TP Elevation Deltas",
    #     [bs[SEG_START_DIST] for bs in base_segments], "Distance (meters)",
    #     [bs[SEG_START_ELEV] 
    #      - bs[SEG_END_ELEV] for bs in base_segments], "Delta Elevation (meters)", "blue"
    # )

    # diagnostic_make_plot(
    #     "Adjacent TP Comparing Distance Techniques",
    #     [bs[SEG_START_DIST] for bs in base_segments],        "Distance",
    #     [bs[SEG_HAVERSINE_DIST] for bs in base_segments],    "Haversine Distance",   "blue",
    #     [bs[SEG_TP_DIST] for bs in base_segments],           "TP Distance",          "red",
    #     [pct_diff(bs[SEG_HAVERSINE_DIST], 
    #               bs[SEG_TP_DIST]) for bs in base_segments], "% Dist Diff",          "green", 
    # )

    # diagnostic_make_plot(
    #     "Elevation and Grade Profile",
    #     [bs[SEG_START_DIST] for bs in base_segments],  "Distance (meters)",
    #     [bs[SEG_START_ELEV] for bs in base_segments], "Elevation (meters)", "blue", 
    #     [bs[SEG_START_ELEV] for bs in base_segments],  "Start Elevation (meters)", "red", 
    # )

    diagnostic_make_plot(
        "Elevation and Grade Profile",
        [bs.start_distance for bs in base_segments],                     "Distance (meters)",
        [bs.start_elevation - bs.end_elevation for bs in base_segments], "Delta Elevation (meters)", "blue", 
        [bs.grade for bs in base_segments],                              "Grade", "green", 
    )

    # diagnostic_make_plot(
    #     "Two Y-Axes Example",
    #     [1, 2, 3, 4, 5],        "X-Axis",
    #     [50, 40, 30, 20, 10],   "Y1 (Blue)",    "blue", 
    #     [5, 15, 25, 35, 45],    "Y2 (Red)",     "red",  
    # )

    # diagnostic_make_plot(
    #     "Three Y-Axes Example",
    #     [ 1,  2,  3,  4,  5],   "X-Axis",
    #     [50, 40, 30, 20, 10],   "Y1 (Blue)",    "red",  
    #     [ 5, 15, 25, 35, 45],   "Y2 (Green)",   "green", 
    #     [ 1,  1,  1,  1,  1],   "Y3 (Orange)",  "orange",
    # )

    # Merge segments by similar grade
    common_grade_segments = merge_segments_by_similar_grade(base_segments)

    # diagnostic_print_list("Common Grade Segments", common_grade_segments) -- too long -- not helpful

    # Summarize the common grade segments
    summarize_common_grade_segments(common_grade_segments)

    
    # Plot the common grade segments
    plot_common_grade_segments(common_grade_segments)

    plt.show()

