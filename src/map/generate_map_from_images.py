import os
from os.path import isfile, join

import gmplot
from dotenv import load_dotenv, find_dotenv


def add_markers(category, categorized_coordinates):
    """
    Adds markers to the map for a given category of images.

    :param categorized_coordinates: A dictionary containing the categorized coordinates.
    :param category: The category of images.
    """
    directory = f'images/categorized/{category}'
    files = [f for f in os.listdir(directory) if isfile(join(directory, f))]
    for file in files:
        if file.endswith('.jpg'):
            # Extract latitude and longitude from the filename
            lat, lng = map(float, file[:-4].split('__'))
            categorized_coordinates[category].append((lat, lng))


def plot_map_from_images():
    """
    Generates a map plot using gmplot with markers and polygons.
    This function creates a map centered around San Francisco, marks a hidden gem,
    highlights some attractions, and outlines the Golden Gate Park.
    """
    # Create the map plotter:
    apikey = os.getenv("GCP_API_KEY")
    start_position = (51.34025362738476, 12.375604273050232)
    gmap = gmplot.GoogleMapPlotter(start_position[0], start_position[1], 13, apikey=apikey)


    categorized_coordinates: dict[int, list] = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
    }
    # Add markers for all categories (1 = best, 6 = worst):
    for category in range(1, 7):
        add_markers(category, categorized_coordinates)



    # Mark a hidden gem:
    gmap.marker(37.770776, -122.461689, color='cornflowerblue')

    # Highlight some attractions:
    attractions_lats, attractions_lngs = zip(*[
        (37.769901, -122.498331),
        (37.768645, -122.475328),
        (37.771478, -122.468677),
        (37.769867, -122.466102),
        (37.767187, -122.467496),
        (37.770104, -122.470436)
    ])
    gmap.scatter(attractions_lats, attractions_lngs, color='#3B0B39', size=40, marker=False)

    # Outline the Golden Gate Park:
    golden_gate_park = zip(*[
        (37.771269, -122.511015),
        (37.773495, -122.464830),
        (37.774797, -122.454538),
        (37.771988, -122.454018),
        (37.773646, -122.440979),
        (37.772742, -122.440797),
        (37.771096, -122.453889),
        (37.768669, -122.453518),
        (37.766227, -122.460213),
        (37.764028, -122.510347)
    ])
    gmap.polygon(*golden_gate_park, color='cornflowerblue', edge_width=10)

    # Draw the map to an HTML file:
    gmap.draw('map.html')

def main():
    load_dotenv(find_dotenv())
    plot_map_from_images()

if __name__ == "__main__":
    main()