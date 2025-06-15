import os
from os.path import isfile, join

import gmplot
from dotenv import load_dotenv, find_dotenv

CATEGORY_COLOR_MAP = {
    1: '#00FF00',  # Best
    2: '#99FF00',  # Good
    3: '#FFFF00',  # Average
    4: '#FF9900',  # Bad
    5: '#FF3300',  # Very Bad
    6: '#FF0000',  # Worst
}

def add_markers(category, categorized_coordinates):
    """
    Adds markers to the map for a given category of images.

    :param categorized_coordinates: A dictionary containing the categorized coordinates.
    :param category: The category of images.
    """

    directory = f'images/sorted_by_class/{category}'
    files = [f for f in os.listdir(directory) if isfile(join(directory, f))]
    for file in files:
        if file.endswith('.jpg'):
            # Extract latitude and longitude from the filename
            lat, lng = file[:-4].split('__')
            lat = float(lat.replace('_', '.'))
            lng = float(lng.replace('_', '.'))

            # Images are categorized from 0 to 5, so we add 1 to the category index
            categorized_coordinates[category+1].append((lat, lng))


def plot_map_from_images():
    """
    Generates a map plot using gmplot with markers and polygons.
    This function creates a map centered around San Francisco, marks a hidden gem,
    highlights some attractions, and outlines the Golden Gate Park.
    """

    apikey = os.getenv("GCP_API_KEY")
    start_position = (51.34025362738476, 12.375604273050232)
    gmap = gmplot.GoogleMapPlotter(start_position[0], start_position[1], 13, apikey=apikey)

    os.chdir("../..")

    categorized_coordinates: dict[int, list] = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
    }
    # Add markers for all categories (1 = best, 6 = worst):
    for category in range(6):
        add_markers(category, categorized_coordinates)

    for category, coordinates in categorized_coordinates.items():
        if coordinates:
            for coordinate in coordinates:
                gmap.circle(coordinate[0], coordinate[1], edge_alpha=0, color=CATEGORY_COLOR_MAP[category], radius=20)

    # Draw the map to an HTML file:
    gmap.draw('map.html')

def main():
    load_dotenv(find_dotenv())
    plot_map_from_images()

if __name__ == "__main__":
    main()