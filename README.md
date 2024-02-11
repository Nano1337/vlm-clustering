# vlm-clustering

First, install git lfs (large file storage): 
```bash
sudo apt update
sudo apt install git
sudo apt install git-lfs
git lfs install
```

Then, clone weights and repo (will take a while depending on network bandwidth) 

```bash 
git clone https://huggingface.co/liuhaotian/llava-v1.6-mistral-7b
```

Apply these changes to the llava repo 
```diff
https://huggingface.co/liuhaotian/llava-v1.6-mistral-7b/commit/edf96c5e9776fdd3f4ef324b5b7831b8b389c440
```


Launch SGLang to serve the model (you'll need to first `pip install sglang[all]`). Be sure to cd a level above the llava repo directory before running the following command.
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path llava-v1.6-mistral-7b --port 30000
```

