import math
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv, find_dotenv

from exceptions import ImageNotFoundException, MetadataNotFoundException
from urlsigner import sign_url


def parse_corners() -> tuple[tuple[str, str], tuple[str, str]]:
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


def get_coordinate_list(
        north_west: tuple[str, str],
        south_east: tuple[str, str],
        step_size_meters=50,
) -> list[tuple[float, float]]:
    """
    Generates a list of coordinates in a grid from north-west to south-east corners.

    :param north_west: Tuple of (latitude, longitude) for the north-west corner.
    :param south_east: Tuple of (latitude, longitude) for the south-east corner.
    :param step_size_meters: The approximate step size in meters for the grid.
    :return: A list of (latitude, longitude) tuples.
    """
    lat_start = float(north_west[0])
    lon_start = float(north_west[1])
    lat_end = float(south_east[0])
    lon_end = float(south_east[1])

    # Generate the coordinates
    coordinates = []

    # Constant: 1 degree of latitude is ~111.32 km
    lat_step = step_size_meters / 111320.0

    current_lat = lat_start

    # Iterate from north to south
    while current_lat >= lat_end:
        # Recalculate longitude step for the current latitude to be accurate
        lon_step = step_size_meters / (111320.0 * math.cos(math.radians(current_lat)))

        current_lon = lon_start
        # Iterate from west to east
        while current_lon <= lon_end:
            coordinates.append((current_lat, current_lon))
            current_lon += lon_step

        current_lat -= lat_step

    return coordinates


def request_street_view_image_metadata(coordinate_lat: str, coordinate_lon: str) -> dict[str, str]:
    """
    Request the Street View image metadata for a given coordinate.
    :param coordinate_lat:
    :param coordinate_lon:
    :return: The pano_id for the given coordinates.
    """
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

    if r.status_code == 200 and r.json()["status"] == "OK":
        json_data = r.json()
        return {"pano_id": json_data["pano_id"], "lat": json_data["location"]["lat"], "lon": json_data["location"]["lng"]}
    else:
        raise MetadataNotFoundException(f"Error: {r.status_code} - {r.text}")


def list_contains_pano_id(data_list: list[tuple[tuple[float, float], str]], target_string: str) -> bool:
    """
    Checks if the target_string is the second item in any of the main tuples in the list.

    :param data_list: A list of tuples, where each tuple is ((float, float), str).
    :param target_string: The string to search for.

    :return: True if the string is found, False otherwise.
    """
    for item_tuple in data_list:
        if item_tuple[1] == target_string:
            return True
    return False


def get_coordinate_pano_ids(contained_coordinates: list[tuple[float, float]]) -> list[tuple[tuple[float, float], str]]:
    """
    Get the pano IDs for the coordinates in the coordinate list.
    :return: A list of tuples with coordinates and their corresponding pano IDs.
    """
    pano_ids_mapped: list[tuple[tuple[float, float], str]] = []
    for current_coordinates in contained_coordinates:
        print(f"Requesting image for location: {current_coordinates}...")
        try:
            image_metadata = request_street_view_image_metadata(str(current_coordinates[0]), str(current_coordinates[1]))
            pano_id = image_metadata.get("pano_id")
            coordinates = (float(image_metadata.get("lat")), float(image_metadata.get("lon")))

            if not list_contains_pano_id(pano_ids_mapped, pano_id):
                pano_ids_mapped.append((coordinates, pano_id))

        except MetadataNotFoundException:
            print(f"Error requesting image for location: {current_coordinates}", file=sys.stderr)
            continue

    print(f"All images requested. {len(pano_ids_mapped)} images found.")

    return pano_ids_mapped


def request_street_view_image_for_id(pano_id: str, image_coordinates: tuple[float, float], path: str = "images/all_images"):
    """
    Request the Street View image for a given pano_id.
    :param pano_id: The pano_id for the image to be requested.
    :param image_coordinates: The coordinates of the image.
    :param path: The path where the image will be saved.

    :return:
    """
    api_key = os.getenv("GCP_API_KEY")
    signing_secret = os.getenv("GCP_SIGNING_SECRET")

    base_url = "https://maps.googleapis.com/maps/api/streetview"

    api_key_param = f"key={api_key}&"
    pano_id_param = f"pano={pano_id}&"
    # 640 x 640 is the maximum size for Street View images on the free tier
    size_param = "size=640x640&"
    pitch_param = "pitch=-90&"
    fov_param = "fov=100&"
    source_param = "source=outdoor&"
    error_code_param = "return_error_code=true&"

    request_url = sign_url(
        input_url=f"{base_url}?{pano_id_param}{size_param}{pitch_param}{fov_param}{source_param}{api_key_param}{error_code_param}",
        secret=signing_secret,
    )

    r = requests.get(request_url)

    if r.status_code == 200:
        save_image_and_set_name(path, r, image_coordinates)
    else:
        raise ImageNotFoundException(f"Error: {r.status_code} - {r.text}")


def encode_coordinates_as_filename(coordinates: tuple[float, float]) -> str:
    """
    Encode the coordinates as a filename.
    :param coordinates: The coordinates to encode.
    :return: A string with the coordinates encoded as a filename.
    """
    latitude = str(coordinates[0]).replace(".", "_")
    longitude = str(coordinates[1]).replace(".", "_")

    return f"{latitude}__{longitude}.jpg"


def save_image_and_set_name(path: str, response: requests.Response, coordinates: tuple[float, float]):
    """
    Save the image and set the name as coordinates.
    :param coordinates: The coordinates of the image.
    :param path: The path where the image will be saved.
    :param response: The api response containing the image data.
    :return:
    """
    # Create the directory if it doesn't exist
    Path(path).mkdir(parents=True, exist_ok=True)

    filename = encode_coordinates_as_filename(coordinates)

    file_path = os.path.join(path, filename)
    with open(file_path, "wb") as f:
        f.write(response.content)


def extract_images():
    """
    Main function to extract images from Google Street View.
    :return:
    """
    load_dotenv(find_dotenv())
    os.chdir("../..")
    input_coordinates = parse_corners()

    all_contained_coordinates = get_coordinate_list(
        input_coordinates[0],
        input_coordinates[1],
    )

    coordinate_pano_ids = get_coordinate_pano_ids(all_contained_coordinates)

    for coordinate, current_pano_id in coordinate_pano_ids:
        print(f"Coordinate: {coordinate}, Pano ID: {current_pano_id}")
        try:
            request_street_view_image_for_id(current_pano_id, coordinate)
        except ImageNotFoundException:
            print(f"Error requesting image for location: {coordinate}", file=sys.stderr)


if __name__ == "__main__":
    extract_images()
