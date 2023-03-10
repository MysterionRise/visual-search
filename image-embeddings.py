import glob
import json
import os
import time
from datetime import datetime

from exif import Image as exifImage
from opensearchpy import OpenSearch
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

host = "localhost"
port = 9200
auth = ("admin", "admin")

DEST_INDEX = "image-embeddings"
CHUNK_SIZE = 200
IMAGE_LIST_SIZE = 200000
MODEL_NAME = "clip-ViT-L-14"

PATH_TO_IMAGES = "images/**/*.jp*g"
PREFIX = "images/"


def bulk(documents):
    actions = []
    for doc in documents:
        index = {"_index": DEST_INDEX}
        action = {"index": index}
        actions.append(action)
        actions.append(doc)

    bulk_data = "\n".join([json.dumps(action) for action in actions]) + "\n"

    return bulk_data


def main():
    lst = []

    start_time = time.perf_counter()
    img_model = SentenceTransformer(MODEL_NAME)
    duration = time.perf_counter() - start_time
    print(f"Duration load model = {duration}")

    filenames = glob.glob(PATH_TO_IMAGES, recursive=True)
    start_time = time.perf_counter()
    for filename in tqdm(
        filenames[:IMAGE_LIST_SIZE],
        desc="Processing files",
        total=len(filenames[:IMAGE_LIST_SIZE]),
    ):
        image = Image.open(filename)

        doc = {}
        embedding = image_embedding(image, img_model)
        doc["image_id"] = create_image_id(filename)
        doc["image_name"] = os.path.basename(filename)
        doc["image_embedding"] = embedding.tolist()
        doc["relative_path"] = os.path.relpath(filename).split(PREFIX)[1]
        doc["exif"] = {}

        try:
            doc["exif"]["date"] = get_exif_date(filename)
            doc["exif"]["location"] = get_exif_location(filename)
        except Exception as e:
            print(e)

        lst.append(doc)

    duration = time.perf_counter() - start_time
    print(f"Duration creating image embeddings = {duration}")

    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_compress=True,  # enables gzip compression for request bodies
        http_auth=auth,
        use_ssl=True,
        verify_certs=False,
    )

    # index name to index data into
    index = DEST_INDEX
    try:
        with open("image-embeddings-mappings.json", "r") as config_file:
            config = json.loads(config_file.read())

            print("Creating index %s" % index)

            index_body = {
                "settings": config["settings"],
                "mappings": config["mappings"],
            }

            response = client.indices.create(index, body=index_body)
            print("\nCreating index:")
            print(response)

        for i in range(0, len(lst), CHUNK_SIZE):
            chunk = lst[i : i + CHUNK_SIZE]
            chunk_actions = bulk(chunk)
            print(client.bulk(chunk_actions))
        duration = time.perf_counter() - start_time
        response = client.indices.flush(index=DEST_INDEX)
        print(response)
        print(f"Total duration = {duration}")
        print("Done!\n")
    except Exception as e:
        print(e)


def image_embedding(image, model):
    return model.encode(image)


def create_image_id(filename):
    # print("Image filename: ", filename)
    return os.path.splitext(os.path.basename(filename))[0]


def get_exif_date(filename):
    with open(filename, "rb") as f:
        image = exifImage(f)
        taken = f"{image.datetime_original}"
        date_object = datetime.strptime(taken, "%Y:%m:%d %H:%M:%S")
        prettyDate = date_object.isoformat()
        return prettyDate


def get_exif_location(filename):
    with open(filename, "rb") as f:
        image = exifImage(f)
        lat = dms_coordinates_to_dd_coordinates(
            image.gps_latitude, image.gps_latitude_ref
        )
        lon = dms_coordinates_to_dd_coordinates(
            image.gps_longitude, image.gps_longitude_ref
        )
        return [lon, lat]


def dms_coordinates_to_dd_coordinates(coordinates, coordinates_ref):
    decimal_degrees = coordinates[0] + coordinates[1] / 60 + coordinates[2] / 3600

    if coordinates_ref == "S" or coordinates_ref == "W":
        decimal_degrees = -decimal_degrees

    return decimal_degrees


if __name__ == "__main__":
    main()
