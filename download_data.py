import os
import requests
from tqdm import tqdm

url = "https://natural-scenes-dataset.s3.amazonaws.com/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
file_path = "nsd_stimuli.hdf5"

# Get the file size from the URL
with requests.get(url, stream=True) as r:
    total_size = int(r.headers.get('content-length', 0))

# Check if file exists to resume download
try:
    file_size = os.path.getsize(file_path)
except OSError:
    file_size = 0

# If the file exists and is complete, skip the download
if file_size >= total_size:
    print("File already downloaded and complete.")
else:
    headers = {"Range": f"bytes={file_size}-"}

    # Adjust total size for the progress bar
    adjusted_size = total_size - file_size

    with requests.get(url, stream=True, headers=headers) as r, open(file_path, "ab") as f, tqdm(
            unit='B',  # unit string to be displayed.
            unit_scale=True,  # let tqdm to determine the scale in kilo, mega..etc.
            unit_divisor=1024,  # define the divisor for calculating the unit scale (1024 for bytes)
            total=adjusted_size,  # the total iteration.
            initial=file_size,  # initial counter value.
            desc=file_path  # prefix for the progress bar.
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)
