import xml.etree.ElementTree as ET
import argparse

def summarize_tcx_file(file_path):
    """
    Summarizes the contents of a TCX file.

    Args:
        file_path (str): Path to the TCX file.

    Prints:
        - TrainingCenterDatabase structure
        - Folders and Courses
        - Course details (name, laps, intensity)
        - Number of trackpoints per track
        - Number of coursepoints
    """
    try:
        # Parse the TCX file
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Define the namespace (TCX files often use a namespace)
        namespace = {'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}

        # TrainingCenterDatabase
        print("TrainingCenterDatabase:")

        # Folders
        folders = root.find('tcx:Folders', namespace)
        if folders is not None:
            print("  Folders:")
            for folder in folders:
                print(f"    - {folder.tag.split('}')[-1]}")  # Remove namespace prefix

        # Courses
        courses = root.find('tcx:Courses', namespace)
        if courses is not None:
            print("  Courses:")
            for course in courses.findall('tcx:Course', namespace):
                # Course name
                course_name = course.find('tcx:Name', namespace)
                print(f"    - Course Name: {course_name.text if course_name is not None else 'Unknown'}")

                # Laps
                laps = course.findall('tcx:Lap', namespace)
                print(f"      Laps: {len(laps)}")
                for lap in laps:
                    intensity = lap.find('tcx:Intensity', namespace)
                    print(f"        - Intensity: {intensity.text if intensity is not None else 'Unknown'}")

                # Tracks
                tracks = course.findall('tcx:Track', namespace)
                print(f"      Tracks: {len(tracks)}")
                for track in tracks:
                    trackpoints = track.findall('tcx:Trackpoint', namespace)
                    print(f"        - Trackpoints: {len(trackpoints)}")

                # CoursePoints
                coursepoints = course.findall('tcx:CoursePoint', namespace)
                print(f"      CoursePoints: {len(coursepoints)}")

    except ET.ParseError as e:
        print(f"Error parsing the TCX file: {e}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Summarize the contents of a TCX file.")
    parser.add_argument(
        "--tcx_file_path",
        type=str,
        default='output/consolidated_tcx_files.tcx',
        help="Path to the TCX file to summarize."
    )
    # parser.add_argument(
    #     "--summary_file_path",
    #     type=str,
    #     default='output/TCX_summary.txt',
    #     help="Path for output"
    # )

    # Parse arguments
    args = parser.parse_args()

    # Call the summarize_tcx_file function
    summarize_tcx_file(args.tcx_file_path)