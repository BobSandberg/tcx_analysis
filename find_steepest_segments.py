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

def diagnostic_make_plot(x, y, xtitle, ytitle, title):

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=title, color="blue")
    plt.xlabel(xtitle, color="blue")
    plt.ylabel(ytitle, color="blue")
    plt.tick_params(axis='y', labelcolor='blue')
    plt.grid(True)
    plt.legend(loc="upper left")

    # plt.show(block=False)


def diagnostic_make_plot_2y(x, y1, y2, xtitle, y1title, y2title, title):

    # Create the figure and primary Y-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot y1
    ax1.plot(x, y1, label="y1Title", color="blue")
    ax1.set_xlabel(xtitle, color="blue")
    ax1.set_ylabel(y1title, color="blue")
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)


    # Plot y2
    ax2 = ax1.twinx()  # Create a secondary y-axis
    ax2.plot(x, y2, label=y2title, color="red")
    ax2.set_ylabel(y2title, color="red")
    # ax2.tick_params(axis='y', labelcolor='red')


    fig.suptitle("2fer")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # plt.show(block=False)



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


def make_plot(base_segments):

    distances, _, _, _, elevations, grades = zip(*base_segments)

    # Create the figure and primary Y-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot elevation profile on the primary Y-axis
    ax1.plot(distances,elevations, label="Elevation Profile", color="blue", marker="o")
    ax1.set_xlabel("Distance (meters)", color="blue")
    ax1.set_ylabel("Elevation (meters)", color="blue")
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)


    # Plot grades on a secondary y-axis
    ax2 = ax1.twinx()  # Create a secondary y-axis
    ax2.plot(distances, grades, label="Grade (%)", color="red")
    ax2.set_ylabel("Grade %", color="red")
    ax2.tick_params(axis='y', labelcolor='red')

    fig.suptitle("Elevation and Grade Profile along route")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.show()


def output_segments(segments):
    """
    Output the segments in two sorted tables, one by average grade and another by latitude/longitude.
    """
    # Table A: Segments Sorted by Average Grade
    print("\nTable A: Segments Sorted by Average Grade")
    sorted_by_grade = sorted(segments, key=lambda x: x["average_grade"], reverse=True)
    table_by_grade = [
        [
            segment["start_point"],
            segment["end_point"],
            f"{segment['average_grade']:.2f}%",
            f"{segment['max_grade']:.2f}%",
            f"{segment['distance']:.2f}m",
        ]
        for segment in sorted_by_grade
    ]
    print(tabulate(table_by_grade, headers=["Start Point", "End Point", "Avg Grade", "Max Grade", "Distance"], tablefmt="grid"))

    # Table B: Segments Sorted by Latitude/Longitude
    print("\nTable B: Segments Sorted by Latitude/Longitude (SE to NW)")
    sorted_by_location = sorted(segments, key=lambda x: (x["start_point"][0], x["start_point"][1]))
    table_by_location = [
        [
            segment["start_point"],
            segment["end_point"],
            f"{segment['average_grade']:.2f}%",
            f"{segment['max_grade']:.2f}%",
            f"{segment['distance']:.2f}m",
        ]
        for segment in sorted_by_location
    ]
    print(tabulate(table_by_location, headers=["Start Point", "End Point", "Avg Grade", "Max Grade", "Distance"], tablefmt="grid"))


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
    # tcx_trackpoint = diagnostic_print_list("tcx_trackpoints", tcx_trackpoints)

    # Parse the TCX file
    trackpoints = extract_data_from_trackpoints(tcx_trackpoints, namespace)
    trackpoints = diagnostic_print_list("extracted trackpoints", trackpoints)
    # diagnostic_make_plot(
    #         [ tp['DistanceMeters'] for tp in trackpoints ],
    #         [ tp['AltitudeMeters'] for tp in trackpoints ], 
    #         'distance (meters)', 'elevation (meters)', 'Elevation Profile'
    # )


    adjacent_trackpoints = window(trackpoints, 2)
    adjacent_trackpoints = diagnostic_print_list("adjacent_trackpoints", adjacent_trackpoints)

    base_segments = compute_base_segments(adjacent_trackpoints)
    base_segments = diagnostic_print_list("base_segments", base_segments) 
            # 0 segment_start_distance_meters,
            # 1 segment_end_distance_meters,
            # 2 segment_haversine_distance_meters,
            # 3 segment_distance_meters,
            # 4 segment_start_elevation_meters,
            # 5 segment_end_elevation_meters,
            # 6 segment_elevation_meters,
            # 7 segment_grade,
    # base_segments = diagnostic_print_list("base_segments", base_segments) 
    x = diagnostic_print_list("base_segments", [ (bs[0], bs[2], bs[3], (bs[2]-bs[3])/bs[2]*100.0) for bs in list(base_segments)[-100:]] ) 
    diagnostic_make_plot(
            [ bs[0] for bs in base_segments ],
            [ (bs[4] - bs[5]) for bs in base_segments ], 
            'distance (meters)', 'delta elevation (meters)', 'Elevation Profile'
    )
    diagnostic_make_plot_2y(
            [ bs[0] for bs in base_segments ],
            [ bs[3] for bs in base_segments ], 
            [ (bs[3] - bs[4]) for bs in base_segments ], 
            'distance', 'haversine', 'haversine vs tp_dist', 'comparing distance techniques'
    )
    diagnostic_make_plot_2y(
        [ bs[0] for bs in base_segments ],
        [ (bs[4] - bs[5]) for bs in base_segments ], 
        [ bs[5] for bs in base_segments ],
        'distance', 'start elev', 'delta elev', 'Elevations'
    )
    diagnostic_make_plot_2y(
        [ bs[0] for bs in base_segments ],
        [ bs[2] for bs in base_segments ], 
        [ (bs[3]-bs[2]) for bs in base_segments ],
        'distance', 'havesine', 'diff', 'Ground distances'
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

