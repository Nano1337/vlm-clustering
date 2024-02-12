# vlm-clustering

Purpose: to see if topic modeling of generated captions from image data allows for more expressivity in clustering compared to clustering by only image embeddings. 

This is part of another project, so I will be using the image portion of the natural scenes dataset (NSD). 

## Run VLM captioning: 

1. To install the NSD image dataset and to download the hdf5 file, please run `download_data.py`, which is set up with the requests library to pick up the download where it left off if the download process was interrupted. Note that this image dataset is ~36GB, which may take a while to download depending on your bandwidth. 

2. To download models, first install git lfs (large file storage): 
```bash
sudo apt update
sudo apt install git
sudo apt install git-lfs
git lfs install
```

3. Then, clone weights and repo (will take a while depending on network bandwidth) 

```bash 
git clone https://huggingface.co/liuhaotian/llava-v1.6-mistral-7b
```

4. Apply these changes to the llava repo (since this model is so new and has bugs)
```diff
https://huggingface.co/liuhaotian/llava-v1.6-mistral-7b/commit/edf96c5e9776fdd3f4ef324b5b7831b8b389c440
```

5. Launch SGLang to serve the model (you'll need to first `pip install sglang[all]`). Be sure to cd a level above the llava repo directory before running the following command.
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path llava-v1.6-mistral-7b --port 30000
```

6. In another terminal, run the `generate_captions.py` script with appropriate modifications to the path and number of samples you want to run caption generation on at the beginning of main. This will save to a parquet file

## Explore Generated Captions with Lilac AI

1. Lilac has a bunch of dependencies so it's best to start with a base installation. Run `pip install lilac` to get started. 

2. Run `lilac start ~/inspect_captions` to launch the interface on a local port. 

3. On the interface, simply name your project and fill in the path to the parquet file. Then, click the purple button to get started. 

## Visualize Embeddings with Nomic

I have a GPU so I'll calculate embeddings locally. Ensure that you have git lfs installed from the first section. 

1. Clone the nomic-embed-text-v1 model: 
```bash 
git clone https://huggingface.co/nomic-ai/nomic-embed-text-v1
```

2. Extract embeddings by running `generate_embeddings.py`

3. I'll eventually bootstrap my own GPU-accelerated pipeline of UMAP dim reduction -> HDBSCAN -> topic modeling visualization as an academic exercise, but for the sake of time I'll just visualize topic clusters with their atlas library. Sign up at nomic.ai and then run `generate_nomic_map.py` with the correct inputs. 

## Corresponding Image data
 
After exploring the Nomic topic clusters, you may want to see the image corresponding to the generated caption. Luckily, there's an id associated with each caption so you can call the `find_img.py` script. For example, if you wanted to see image 345, then run: 
```bash
python find_img.py --id 345
```

and the output will be shown in `figs/image.png`