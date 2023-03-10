import os
import random
import time

from opensearchpy import OpenSearch
from PIL import Image
from sentence_transformers import SentenceTransformer

host = "localhost"
port = 9200
auth = ("admin", "admin")
PATH_TO_IMAGES = "images/"
DEST_INDEX = "image-embeddings"


def image_embedding(image, model):
    return model.encode(image)


def main():
    start_time = time.perf_counter()
    img_model = SentenceTransformer("clip-ViT-B-32")
    duration = time.perf_counter() - start_time
    print(f"Duration load model = {duration}")

    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_compress=True,  # enables gzip compression for request bodies
        http_auth=auth,
        use_ssl=True,
        verify_certs=False,
    )

    filenames = os.listdir(PATH_TO_IMAGES)
    random_filename = random.choice(filenames)
    # random_filename = "5TpBhNBPAE8.jpg"
    with Image.open(PATH_TO_IMAGES + random_filename) as image:
        # Show the image using Pillow
        image.show()
        embedding = image_embedding(image, img_model)
        query = {
            "size": 5,
            "query": {
                "knn": {"image_embedding": {"vector": embedding.tolist(), "k": 5}}
            },
        }
        print(query)

        response = client.search(body=query, index=DEST_INDEX)

        print("\nSearch results:")
        hits = response["hits"]["hits"]
        for hit in hits:
            res_name = hit["_source"]["image_name"]
            with Image.open(PATH_TO_IMAGES + res_name) as res_image:
                # Show the image using Pillow
                res_image.show()


if __name__ == "__main__":
    main()
