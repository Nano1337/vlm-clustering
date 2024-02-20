import os
import h5py
from tqdm import tqdm
import tempfile
from PIL import Image
import time
import concurrent.futures
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from fastparquet import write
from datasets import load_dataset

import sglang as sgl
from sglang import set_default_backend, RuntimeEndpoint

import time

set_default_backend(RuntimeEndpoint("http://localhost:30000"))

@sgl.function
def image_qa(s, image_path, question, regex=None): 
    s += sgl.user(sgl.image(image_path) + question)
    s += sgl.assistant(sgl.gen("answer", regex=regex))

def make_dataset(image_dir): 
    dataset = load_dataset("MMVP/MMVP")
    images = dataset['train']['image']
    for i, image in enumerate(images): 
        image.save(os.path.join(image_dir, f"{i}.png"))

# FIXME: the huggingface dataset is out of order. Redownload and resave image directory once they fix it
def visualize_output(question, response, row): 
    print(question)
    print(response)
    print(f'Correct Answer: {row["Correct Answer"]}')

if __name__ == "__main__": 

    image_dir = "images/"

    # fetch images if you don't have it already
    if not os.path.exists(image_dir): 
        os.makedirs(image_dir)
        make_dataset(image_dir)
    elif len(os.listdir(image_dir)) == 0: 
        make_dataset(image_dir)

    df = pd.read_csv('Questions.csv')
    outputs = []
    for index, row in tqdm(df.iterrows()): 
        question = "You must answer in format \"(a)\" or \"(b)\". " + row['Question'] + " " + row['Options']
 
        image_path = os.path.join(image_dir, f"{index}.png")

        response = image_qa.run(
            image_path = image_path, 
            # question = "Generate a complex description of the image",
            question = question,
            max_new_tokens=4096, 
        )["answer"]

        # Optional: visualize output
        visualize_output(question, response, row)
        # exit()

        new_row = {'index': index, 'gt': row['Correct Answer'], 'answer': response}
        outputs.append(new_row)

    output_df = pd.DataFrame(outputs)
    output_df.to_csv('output.csv', index=False)