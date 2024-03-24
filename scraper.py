import argparse
from dotenv import load_dotenv
import requests
from requests import exceptions
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True, help="Search Bing Image API with query")
ap.add_argument("-o", "--output", required=True, help="path to output directory of images")
args = vars(ap.parse_args())

load_dotenv()

API_KEY = os.getenv("BING_API_KEY")
SEARCH_URL = "https://api.bing.microsoft.com/v7.0/images/search"
MAX_RESULTS = 250
GROUP_SIZE = 50

EXCEPTIONS = set([IOError, FileNotFoundError,
	exceptions.RequestException, exceptions.HTTPError,
	exceptions.ConnectionError, exceptions.Timeout])

# setting search params
query = args["query"]
headers = {"Ocp-Apim-Subscription-Key": API_KEY}
params = {"q": query, "offset": 0, "count": GROUP_SIZE} 

# searching for images
print("[INFO] searching Bing API for '{}'".format(query))
response = requests.get(SEARCH_URL, headers=headers, params=params)
response.raise_for_status()

# printing image results
results = response.json()
est_num_results = min(results["totalEstimatedMatches"], MAX_RESULTS)
print("[INFO] {} total results for '{}'".format(est_num_results, query))

# saving images
total = 0
for offset in range(0, est_num_results, GROUP_SIZE):
    print("[INFO] making request for group {}-{} of {} ...".format(offset, offset + GROUP_SIZE, est_num_results))
    
    params["offset"] = offset
    response = requests.get(SEARCH_URL, headers=headers, params=params)
    response.raise_for_status()
    results = response.json()
    print("[INFO] saving images for group {}-{} of {} ...".format(offset, offset + GROUP_SIZE, est_num_results))

    # iterate over results
    for v in results["value"]:
        try: #downloading image
            # making request to download image
            print("[INFO] fetching: {}".format(v["contentUrl"]))
            re = requests.get(v["contentUrl"], timeout=30)

            # build the path to output image
            extension = v["contentUrl"][v["contentUrl"].rfind("."):]

            if (extension == ".jpg") or (extension == ".png"):
                path = os.path.sep.join([args["output"], "{}{}".format(str(total).zfill(7), extension)])
            else:
                continue

            # writing data to disk
            file = open(path, "wb")
            file.write(re.content)
            file.close()

        except Exception as e:
            if type(e) in EXCEPTIONS:
                print(e)
                print("[INFO] skipping: {}".format(v["contentUrl"]))
                continue
        image = cv2.imread(path)

        if image is None:
            print("[INFO] deleting: {}".format(path))
            os.remove(path)
            continue

        total += 1
