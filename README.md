# visual-search
Visual Search approach for OpenSearch

### How to run

1. Install dependencies by running `pip install -r requirements.txt`
2. Run Opensearch instance via Docker by running `docker-compose -f docker-compose.yml up`
3. Put images into `images` or download available datasets, such as there - https://huggingface.co/datasets/jamescalam/unsplash-25k-photos
4. Run `python image-embeddings.py`
5. Open Jupyter Notebook - `visual-search-demo.ipynb` play with it on visual search. You could supply image into embedding or text to find similar image
6. Enjoy!
