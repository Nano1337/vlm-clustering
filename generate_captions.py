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

import sglang as sgl
from sglang import set_default_backend, RuntimeEndpoint

import time

set_default_backend(RuntimeEndpoint("http://localhost:30000"))

@sgl.function
def image_qa(s, image_path, question, regex=None): 
    s += sgl.user(sgl.image(image_path) + question)
    s += sgl.assistant(sgl.gen("answer", regex=regex))

def save_image(image, index):
    """
    Saves an image to a temporary file and returns the file path.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        Image.fromarray(image).save(temp_file.name)
        return temp_file.name

def delete_file(file_path):
    """
    Deletes a file given its file path.
    """
    os.remove(file_path)

def append_batch_to_parquet(df, parquet_file):
    if os.path.exists(parquet_file):
        write(parquet_file, df, append=True, compression='snappy')
    else:
        write(parquet_file, df, compression='snappy')
    
    print("Added to parquet file")

if __name__ == "__main__":
    """
    We have to create temporary file paths since that's what the sglang function expects.
    Note: that I will only process 1k out of the 73k images in the dataset due to time constraints.
    """

    file_path = "/home/haoli/Documents/data/nsd_images/nsd_stimuli.hdf5"
    parquet_file = "captions.parquet"
    captions = []
    counter = 0
    with h5py.File(file_path, 'r') as f:
 
        # # for 1000 total samples should take about an hour
        for i in tqdm(range(100)):

            # manually set chunk size to 10 for now
            chunk_size = 10
            
            data = f['imgBrick'][counter:counter+chunk_size]

            # Parallelize the saving of images to temporary files
            temp_file_paths = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_image = {executor.submit(save_image, image, i): i for i, image in enumerate(data)}
                for future in concurrent.futures.as_completed(future_to_image):
                    temp_file_path = future.result()
                    temp_file_paths.append(temp_file_path)

            # generate captions for each image
            for i, file_path in enumerate(temp_file_paths): 

                response = image_qa.run(
                    image_path = file_path, 
                    question = "Generate a complex description of the image",
                    max_new_tokens=4096, 
                )["answer"]

                captions.append(response)

            # Parallelize the deletion of temporary files
            with concurrent.futures.ThreadPoolExecutor() as executor:
                delete_tasks = [executor.submit(delete_file, file_path) for file_path in temp_file_paths]

                # error handling
                for future in concurrent.futures.as_completed(delete_tasks):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error deleting file: {e}")

            data = {
                'id': [str(i).zfill(5) for i in range(counter, counter + chunk_size)],
                'caption': captions
            }

            df = pd.DataFrame(data)
            append_batch_to_parquet(df, parquet_file)

            captions = []
            counter += chunk_size
