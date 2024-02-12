from nomic import atlas
from nomic import data_inference
import pandas as pd
import numpy as np
from fastparquet import ParquetFile

pf = ParquetFile("captions_ordered.parquet")
df = pf.to_pandas()
embeddings = np.load('embeddings.npy')

# Note: Working Demo: https://atlas.nomic.ai/data/haoliyin/lazy-jemison/map

topic_model = data_inference.NomicTopicOptions(build_topic_model=True, topic_label_field="caption")

dataset = atlas.map_data(
    data=df, 
    embeddings=embeddings, 
    id_field='id', 
    topic_model=topic_model, 
)


