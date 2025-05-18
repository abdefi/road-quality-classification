import os

import requests
from dotenv import load_dotenv, find_dotenv
from urlsigner import sign_url


def init():
    load_dotenv(find_dotenv())


def parse_corners():
    """
    Parse the corners of the bounding box from environment variables.
    :return: Tuple of (north_west, south_east)
    """
    north_west = os.getenv("NORTH_WEST").split(",")
    south_east = os.getenv("SOUTH_EAST").split(",")

    if len(north_west) != 2 or len(south_east) != 2:
        raise ValueError("NORTH_WEST and SOUTH_EAST must be in the format 'lat,lon'")

    print(f"Parsed Coordinates: {north_west}, {south_east}")

    return (north_west[0], north_west[1]), (south_east[0], south_east[1])


def get_coordinate_list(north_west: tuple[str, str], south_east: tuple[str, str], step_size_meters=10):
    """
    Generate a list of coordinates between two points.
    :param north_west: Tuple of (latitude, longitude) for the north-west corner
    :param south_east: Tuple of (latitude, longitude) for the south-east corner
    :param step_size_meters: Step size in meters
    :return: List of tuples with coordinates
    """
    # Convert step size from meters to degrees (approximate)
    lat_step = step_size_meters / 111320  # 1 degree latitude is approximately 111.32 km
    lon_step = step_size_meters / (111320 * abs(float(north_west[0]) - float(south_east[0])))

    lat_start = float(north_west[0])
    lon_start = float(north_west[1])
    lat_end = float(south_east[0])
    lon_end = float(south_east[1])

    difference_lat = lat_start - lat_end
    difference_lon = lon_start - lon_end

    # Calculate the number of steps needed
    steps_lat = int(abs(difference_lat) / lat_step)
    steps_lon = int(abs(difference_lon) / lon_step)

    # Generate the coordinates
    coordinates = []
    for i in range(steps_lat + 1):
        for j in range(steps_lon + 1):
            lat = lat_start - (i * lat_step)
            lon = lon_start - (j * lon_step)
            coordinates.append((lat, lon))

    return coordinates


def request_street_view_image(coordinate_lat: str, coordinate_lon: str):
    api_key = os.getenv("GCP_API_KEY")
    signing_secret = os.getenv("GCP_SIGNING_SECRET")

    base_url = "https://maps.googleapis.com/maps/api/streetview"

    api_key_param = f"key={api_key}&"
    metadata_param = f"metadata?location={coordinate_lat},{coordinate_lon}&"

    request_url = sign_url(
        input_url=f"{base_url}/{metadata_param}{api_key_param}",
        secret=signing_secret,
    )

    r = requests.get(request_url)
    print(r.text)


if __name__ == "__main__":
    init()
    input_coordinates = parse_corners()

    coordinate_list = get_coordinate_list(
        input_coordinates[0],
        input_coordinates[1],
    )

    for coordinate in coordinate_list:
        print(f"Requesting image for location: {coordinate}...")
        request_street_view_image(str(coordinate[0]), str(coordinate[1]))
