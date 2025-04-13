import os
import re
import xml.etree.ElementTree as ET

def get_filtered_and_sorted_files(input_directory, pattern, sort_key_function):
    """
    Filters and sorts TCX files in the input directory based on a pattern and sort key function.
    """
    # Get all files in the directory
    all_files = os.listdir(input_directory)

    # Filter files based on the provided pattern
    filtered_files = [f for f in all_files if re.match(pattern, f)]

    # Sort the filtered files using the provided sort key function
    sorted_files = sorted(filtered_files, key=sort_key_function)

    return [os.path.join(input_directory, f) for f in sorted_files]


def consolidate_tcx_files(filtered_sorted_files, output_file):
    """
    Consolidates the filtered and sorted TCX files into a single TCX file.
    """
    # Namespaces for TCX files
    namespaces = {'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}

    # Create a root element for the consolidated file
    consolidated_root = ET.Element('TrainingCenterDatabase', xmlns=namespaces['tcx'])

    # Loop through the filtered and sorted TCX files
    for file_path in filtered_sorted_files:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Find all activities and append them to the consolidated root
        activities = root.find('tcx:Activities', namespaces)
        if activities is not None:
            consolidated_activities = consolidated_root.find('Activities')
            if consolidated_activities is None:
                consolidated_activities = ET.SubElement(consolidated_root, 'Activities')
            for activity in activities:
                consolidated_activities.append(activity)

    # Write the consolidated TCX file
    tree = ET.ElementTree(consolidated_root)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Consolidated TCX file created: {output_file}")


def summarize_tcx_file(tcx_file):
    """
    Reads a consolidated TCX file and outputs a summary of the activities.
    """
    # Namespaces for TCX files
    namespaces = {'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}

    # Parse the TCX file
    tree = ET.parse(tcx_file)
    root = tree.getroot()

    # Extract and summarize activity data
    summary = []
    activities = root.find('tcx:Activities', namespaces)
    if activities is not None:
        for activity in activities:
            sport = activity.attrib.get('Sport', 'Unknown')
            start_time = activity.find('tcx:Id', namespaces).text
            laps = activity.findall('tcx:Lap', namespaces)
            total_time = sum(float(lap.find('tcx:TotalTimeSeconds', namespaces).text) for lap in laps)
            total_distance = sum(float(lap.find('tcx:DistanceMeters', namespaces).text) for lap in laps)
            summary.append({
                'Sport': sport,
                'Start Time': start_time,
                'Total Time (seconds)': total_time,
                'Total Distance (meters)': total_distance
            })

    # Print the summary
    for i, activity in enumerate(summary, 1):
        print(f"Activity {i}:")
        for key, value in activity.items():
            print(f"  {key}: {value}")
        print()

    return summary


if __name__ == "__main__":
    # Specify the input directory containing TCX files
    input_directory = "tcx_files"

    # Specify the output consolidated TCX file
    output_file = "consolidated.tcx"

    # Define the pattern to match filenames (e.g., files with "RoutePart" in the name)
    pattern = r".*RoutePart\d+.*\.tcx"

    # Define the sort key function (e.g., extract the route part number from the filename)
    def sort_key_function(filename):
        match = re.search(r"RoutePart(\d+)", filename)
        return int(match.group(1)) if match else float('inf')  # Sort by route part number, or push unmatched files to the end

    # Get the filtered and sorted list of TCX files
    filtered_sorted_files = get_filtered_and_sorted_files(input_directory, pattern, sort_key_function)

    # Print filenames with index numbers
    [print(f"{i + 1}. {filename}") for i, filename in enumerate(filtered_sorted_files)]

    # Consolidate the filtered and sorted TCX files
    consolidate_tcx_files(filtered_sorted_files, output_file)

    # Summarize the consolidated TCX file
    summary = summarize_tcx_file(output_file)

    # Output the summary in a shareable format
    print("Summary of activities (shareable):")
    for activity in summary:
        print(activity)
