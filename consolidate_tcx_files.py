import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from itertools import tee, islice
from tabulate import tabulate
import argparse
from enum import Enum


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

# ----------------------------------

def create_sort_key_function(sortkey_extraction_pattern):
    """
    Creates a sort key function that extracts a numeric key from a filename
    using the provided regex pattern.
    """
    def sort_key_function(filename):
        match = re.search(sortkey_extraction_pattern, filename)
        return int(match.group(1)) if match else float('inf')  # Sort by route part number
    return sort_key_function



def get_filtered_and_sorted_files(input_directory, filter_pattern, sort_key_function):
    """
    Filters and sorts TCX files in the input directory based on a pattern and sort key function.
    """
    # Get all files in the directory
    all_files = os.listdir(input_directory)
    diagnostic_print_list("All files in directory", all_files)

    # Filter files based on the provided pattern
    filtered_files = [f for f in all_files if re.match(filter_pattern, f)]
    diagnostic_print_list("Filtered files matching pattern", filtered_files)

    # Sort the filtered files using the provided sort key function
    sorted_files = sorted(filtered_files, key=sort_key_function)
    diagnostic_print_list("Sorted filtered files", sorted_files)

    return [os.path.join(input_directory, f) for f in sorted_files]


class ConsolidationMode(Enum):
    ONE_CONSOLIDATED_COURSE = "one_consolidated_course"
    LIST_OF_COURSES = "list_of_courses"

def consolidate_as_one_course(root, filtered_sorted_files, namespace):
    """
    Consolidates multiple TCX files into one single course by appending all trackpoints
    to the first course's track.
    """
    # Find the Courses node in the first file
    courses_node = root.find("ns:Courses", namespace)
    if courses_node is None:
        raise ValueError("Courses node not found in the first file.")

    # Iterate over the remaining files and append their course data to the first course
    for file_path in filtered_sorted_files[1:]:
        print(f"Processing file: {file_path}")
        tree = ET.parse(file_path)
        file_root = tree.getroot()
        file_courses = file_root.find("ns:Courses", namespace)

        if file_courses is not None:
            for course in file_courses:
                for track in course.findall("ns:Track", namespace):
                    # Append all trackpoints to the first course's track
                    first_course_track = courses_node[0].find("ns:Track", namespace)
                    if first_course_track is None:
                        raise ValueError("Track node not found in the first course of the first file.")
                    for trackpoint in track:
                        first_course_track.append(trackpoint)


def consolidate_as_list_of_courses(root, filtered_sorted_files, namespace):
    """
    Consolidates multiple TCX files into a list of separate courses.
    """
    # Find the Courses node in the first file
    courses_node = root.find("ns:Courses", namespace)
    if courses_node is None:
        raise ValueError("Courses node not found in the first file.")

    # Iterate over the remaining files and append their courses as separate entries
    for file_path in filtered_sorted_files[1:]:
        print(f"Processing file: {file_path}")
        tree = ET.parse(file_path)
        file_root = tree.getroot()
        file_courses = file_root.find("ns:Courses", namespace)

        if file_courses is not None:
            for course in file_courses:
                courses_node.append(course)


def consolidate_tcx_files(filtered_sorted_files, output_path, mode=ConsolidationMode.LIST_OF_COURSES):
    """
    Consolidates multiple TCX files into a single TCX file.

    Args:
        filtered_sorted_files (list): List of file paths to the TCX files to consolidate.
        output_path (str): Path to the output consolidated TCX file.
        mode (ConsolidationMode): Determines whether to create one consolidated course or a list of courses.
    """
    if not filtered_sorted_files:
        print("No files to consolidate.")
        return

    # Parse the first file to get the root structure
    first_file = filtered_sorted_files[0]
    tree = ET.parse(first_file)
    root = tree.getroot()

    # Namespace for TCX files
    namespace = {"ns": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}

    # Call the appropriate consolidation function based on the mode
    if mode == ConsolidationMode.ONE_CONSOLIDATED_COURSE:
        consolidate_as_one_course(root, filtered_sorted_files, namespace)
    elif mode == ConsolidationMode.LIST_OF_COURSES:
        consolidate_as_list_of_courses(root, filtered_sorted_files, namespace)
    else:
        raise ValueError(f"Invalid consolidation mode: {mode}")

    # Write the consolidated TCX file
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Consolidated TCX file written to: {output_path}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Consolidate and summarize TCX files.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default='tcx_files',
        help="Path to the input directory containing TCX files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='output',
        help="Path to the output directory for the consolidated TCX file."
    )
    parser.add_argument(
        "--filter_patt",
        type=str,
        default=r"LC_.*_Main.*.tcx",
        help="Regex pattern to filter filenames (default: 'LC_.*_Main.*.tcx')."
    )
    parser.add_argument(
        "--sortkey_extraction_patt",
        type=str,
        default=r"LC_(\d+)",
        help="Regex pattern to extract the sort key from filenames (default: 'LC_(\d+)')."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[mode.value for mode in ConsolidationMode],
        default=ConsolidationMode.ONE_CONSOLIDATED_COURSE.value,
        help="Consolidation mode: 'one_consolidated_course' or 'list_of_courses' (default: 'list_of_courses')."
    )

    # Parse arguments
    args = parser.parse_args()

    # Assign arguments to variables
    input_dir = args.input_dir
    output_dir = args.output_dir
    filter_pattern = args.filter_patt
    sortkey_extraction_pattern = args.sortkey_extraction_patt
    mode = ConsolidationMode(args.mode)  # Convert the string back to the enum

    # Specify the output consolidated TCX file
    output_path = f"{output_dir}/consolidated_tcx_files.tcx"

    # Get the filtered and sorted list of TCX files
    filtered_sorted_files = get_filtered_and_sorted_files(input_dir, filter_pattern, create_sort_key_function(sortkey_extraction_pattern))
    diagnostic_print_list("Filtered and sorted TCX files", filtered_sorted_files)

    # Consolidate the filtered and sorted TCX files
    consolidate_tcx_files(filtered_sorted_files, output_path, mode=mode)
