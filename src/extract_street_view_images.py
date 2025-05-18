import os

import requests
from dotenv import load_dotenv, find_dotenv
from urlsigner import sign_url

load_dotenv(find_dotenv())

api_key = os.getenv("GCP_API_KEY")
signing_secret = os.getenv("GCP_SIGNING_SECRET")

base_url = "https://maps.googleapis.com/maps/api/streetview"
location = "51.338114,12.381790"
api_key_param = f"key={api_key}&"
metadata_param = f"metadata?location={location}&"


request_url = sign_url(
    input_url=f"{base_url}/{metadata_param}{api_key_param}",
    secret=signing_secret,
)

r = requests.get(request_url)
print(r.text)