import xml.etree.ElementTree as ET
from math import radians, sin, cos, sqrt, atan2
import argparse
import matplotlib.pyplot as plt
from collections.abc import Iterator
from itertools import tee, islice
from tabulate import tabulate


def diagnostic_print_list(title, items):
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

def diagnostic_make_plot(window_title, x, x_title, *args):
    """
    Create a plot with one X-axis and multiple Y-axes.

    Args:
        window_title (str): Title for the plot window/tab.
        x (list): The X-axis data.
        x_title (str): Label for the X-axis.
        *args: Alternating Y-axis data, color, and title (e.g., y1, y1_color, y1_title, y2, y2_color, y2_title, ...).
    """
    if len(args) % 3 != 0:
        raise ValueError("Each Y-axis data series must have a corresponding color and title.")

    # Create the figure and the first Y-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.manager.set_window_title(window_title)

    # Plot the first Y-axis
    y1, y1_color, y1_title = args[0], args[1], args[2]
    ax1.plot(x, y1, label=y1_title, color=y1_color)
    ax1.set_xlabel(x_title, color="black")
    ax1.set_ylabel(y1_title, color=y1_color)
    ax1.tick_params(axis='y', labelcolor=y1_color)
    ax1.grid(True)

    # Add additional Y-axes if needed
    axes = [ax1]
    for i in range(3, len(args), 3):
        y_data = args[i]
        y_color = args[i + 1]
        y_title = args[i + 2]
        ax = ax1.twinx()  # Create a new Y-axis
        ax.spines['right'].set_position(('outward', 60 * ((i // 3) - 1)))  # Offset each additional Y-axis
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
    Between each two segments compute the delta values

    Args:
        trackpoints

    Returns:

    """
    return [ compute_trackpoint_deltas(tp1,tp2) for tp1,tp2 in adjacent_trackpoints ]


# # Process trackpoints to find steepest segments
#def find_steepest_segments(trackpoints, namespace, min_distance=50, grade_delta=5, min_grade=7):
#     segments = []
#     grades = []  # Store grades for all segments
#     haversine_distances = []  # Store haversine distances for all segments
#     dm_distance_meters = []  # Store DistanceMeters values from the TCX file
#     i = 0


#     while i < len(trackpoints):
#         cumulative_haversine_distance = 0
#         cumulative_dm_distance = 0
#         cumulative_altitude_change = 0
#         max_grade = 0
#         start_point = trackpoints[i]
#         altitude_start = float(start_point.find("ns:AltitudeMeters", namespace).text)
#         lat_start = float(start_point.find("ns:Position/ns:LatitudeDegrees", namespace).text)
#         lon_start = float(start_point.find("ns:Position/ns:LongitudeDegrees", namespace).text)

#         # Extract DistanceMeters for the starting point
#         tmp = start_point.find("ns:DistanceMeters", namespace)
#         if tmp is not None:
#             dm_distance_start = float(tmp.text)
#         else:
#             dm_distance_start = 0

#         for j in range(i + 1, len(trackpoints)):
#             end_point = trackpoints[j]
#             altitude_end = float(end_point.find("ns:AltitudeMeters", namespace).text)
#             lat_end = float(end_point.find("ns:Position/ns:LatitudeDegrees", namespace).text)
#             lon_end = float(end_point.find("ns:Position/ns:LongitudeDegrees", namespace).text)

#             haversine_distance = haversine(lat_start, lon_start, lat_end, lon_end)
#             cumulative_haversine_distance += haversine_distance
#             delta_altitude = altitude_end - altitude_start
#             grade = (delta_altitude / cumulative_haversine_distance) * 100 if cumulative_haversine_distance > 0 else 0
#             max_grade = max(max_grade, abs(grade))

#             # Store the grade and haversine distance for this segment
#             grades.append(grade)
#             haversine_distances.append(cumulative_haversine_distance)

#             # Extract DistanceMeters for the ending point
#             dm_distance_end = end_point.find("ns:DistanceMeters", namespace)
#             if dm_distance_end is not None:
#                 dm_distance_end = float(dm_distance_end.text)
#             else:
#                 dm_distance_end = 0

#             dm_distance_meters.append(dm_distance_end - dm_distance_start)

#             if cumulative_haversine_distance >= min_distance:
#                 if segments and abs(segments[-1]["average_grade"] - grade) <= grade_delta:
#                     segments[-1]["end_point"] = (lat_end, lon_end)
#                     segments[-1]["distance"] += cumulative_haversine_distance
#                     segments[-1]["altitude_change"] += delta_altitude
#                     segments[-1]["max_grade"] = max(segments[-1]["max_grade"], max_grade)
#                     segments[-1]["average_grade"] = (segments[-1]["altitude_change"] / segments[-1]["distance"]) * 100
#                 else:
#                     if grade >= min_grade:  # Only add segments that meet the minimum grade
#                         segments.append({
#                             "start_point": (lat_start, lon_start),
#                             "end_point": (lat_end, lon_end),
#                             "distance": cumulative_haversine_distance,
#                             "altitude_change": delta_altitude,
#                             "average_grade": grade,
#                             "max_grade": max_grade,
#                         })
#                 break

#         i += 1

#     return segments, grades, haversine_distances, dm_distance_meters


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
    trackpoints = diagnostic_print_list("Extracted Trackpoints", trackpoints)

    diagnostic_make_plot(
        "Trackpoint Elevation Profile",
        [tp['DistanceMeters'] for tp in trackpoints],
        "Distance (meters)",
        [tp['AltitudeMeters'] for tp in trackpoints], "blue", "Elevation (meters)"
    )

    adjacent_trackpoints = window(trackpoints, 2)
    adjacent_trackpoints = diagnostic_print_list("Adjacent Trackpoints", adjacent_trackpoints)

    base_segments = compute_base_segments(adjacent_trackpoints)
    base_segments = diagnostic_print_list("Base Segments", base_segments)

    # Easy Reference to Tuple positions for base_segments Tuple
    seg_start_distance = 0
    seg_end_distance = 1
    seg_haversine_distance = 2
    seg_tp_distance = 3
    seg_start_elevation = 4
    seg_end_elevation = 5
    seg_elevation = 6
    seg_grade = 7

    diagnostic_make_plot(
        "Adjacent TP Elevation Deltas",
        [bs[seg_start_distance] for bs in base_segments],   "Distance (meters)",
        [bs[seg_start_elevation] 
         - bs[seg_end_elevation] for bs in base_segments],  "blue", "Delta Elevation (meters)"
    )

    diagnostic_make_plot(
        "Adjacent TP Comparing Distance Techniques",
        [bs[seg_start_distance] for bs in base_segments],           "Distance",
        [bs[seg_haversine_distance] for bs in base_segments],       "blue", "Haversine Distance",
        [bs[seg_tp_distance] for bs in base_segments],              "red", "TP Distance",
        [pct_diff(bs[seg_haversine_distance], 
                  bs[seg_tp_distance]) for bs in base_segments],    "green", "% Dist Diff"
    )

    diagnostic_make_plot(
        "Elevation and Grade Profile",
        [bs[seg_start_distance] for bs in base_segments],                           "Distance (meters)",
        [bs[seg_start_elevation] - bs[seg_end_elevation] for bs in base_segments],  "blue", "Delta Elevation (meters)",
        [bs[seg_start_elevation] for bs in base_segments],                          "red", "Start Elevation (meters)"
    )

    diagnostic_make_plot(
        "Two Y-Axes Example",
        [1, 2, 3, 4, 5],                "X-Axis",
        [50, 40, 30, 20, 10],   "blue", "Y1 (Blue)",
        [5, 15, 25, 35, 45],    "red",  "Y2 (Red)"
    )

    diagnostic_make_plot(
        "Three Y-Axes Example",
        [ 1,  2,  3,  4,  5],            "X-Axis",
        [50, 40, 30, 20, 10],   "blue",  "Y1 (Blue)",
        [ 5, 15, 25, 35, 45],   "green", "Y2 (Green)",
        [ 1,  1,  1,  1,  1],   "orange","Y3 (Orange)"
    )

    plt.show()


#     # Find steepest segments
#     segments, grades, haversine_distances, distance_meters = find_steepest_segments(
#         trackpoints, namespace,
#         min_distance=args.min_distance,
#         grade_delta=args.grade_delta,
#         min_grade=args.min_grade
#     )

#     # Output the results
#     output_segments(segments)

#     # Thresholds
#     MAX_GRADE_THRESHOLD = 25  # in percentage
#     DISTANCE_DIFF_THRESHOLD = 0.02  # 2% difference

#     # Extract elevations, latitudes, and longitudes using list comprehensions
#     elevations = [float(tp.find("ns:AltitudeMeters", namespace).text) for tp in trackpoints]
#     latitudes = [float(tp.find("ns:Position/ns:LatitudeDegrees", namespace).text) for tp in trackpoints]
#     longitudes = [float(tp.find("ns:Position/ns:LongitudeDegrees", namespace).text) for tp in trackpoints]

#     # Iterate through the data points
#     min_length = min(len(grades), len(haversine_distances), len(distance_meters), len(latitudes), len(longitudes))

#     questionable_grades = [
#         (i, grades[i], latitudes[i], longitudes[i])
#         for i in range(min_length)
#         if grades[i] > MAX_GRADE_THRESHOLD
#     ]

#     questionable_distances = [
#         (
#             i,
#             haversine_distances[i],
#             distance_meters[i],
#             abs(haversine_distances[i] - distance_meters[i]) / distance_meters[i] * 100,
#             latitudes[i],
#             longitudes[i],
#         )
#         for i in range(min_length)
#         if abs(haversine_distances[i] - distance_meters[i]) / distance_meters[i] > DISTANCE_DIFF_THRESHOLD
#     ]

#     # Log questionable grades
#     if questionable_grades:
#         print("Questionable Grades (Index, Grade, Latitude, Longitude):")
#         for index, grade, lat, lon in questionable_grades:
#             print(f"Index: {index}, Grade: {grade:.2f}%, Latitude: {lat:.6f}, Longitude: {lon:.6f}")

#     # Write questionable distances to a file
#     if questionable_distances:
#         with open("questionable_distances.txt", "w") as f:
#             f.write("Index, Haversine Distance, DistanceMeters, Percent Difference, Latitude, Longitude\n")
#             for index, haversine_distance, distance_meters, percent_diff, lat, lon in questionable_distances:
#                 f.write(f"{index}, {haversine_distance:.2f}, {distance_meters:.2f}, {percent_diff:.2f}%, {lat:.6f}, {lon:.6f}\n")
#     print("\nQuestionable distances have been written to 'questionable_distances.txt'.")

    # make_plot(base_segments)