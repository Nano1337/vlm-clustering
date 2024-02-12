from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel 
from fastparquet import ParquetFile

# load models
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', model_max_length=8192)
model = AutoModel.from_pretrained('./nomic-embed-text-v1/', trust_remote_code=True, rotary_scaling_factor=2)
model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

if __name__ == "__main__": 

    pf = ParquetFile("captions.parquet")
    df = pf.to_pandas()
    
    # read in batches of 10 
    counter = 0
    batch_size = 50
    num_batches = int(1000/batch_size) # must be an int
    embeddings_list = []
    for i in tqdm(range(num_batches)): 

        # convert batch to a list with "clustering: " appended to the front of each phrase
        sentences = df['caption'][counter : counter + batch_size].tolist()
        sentences = ["clustering: " + s for s in sentences]

        # tokenize
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # extract embeddings
        with torch.no_grad(): 
            model_output = model(**encoded_input)
            embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # add to numpy output array 
        embeddings_np = embeddings.cpu().numpy()
        embeddings_list.append(embeddings_np)

        counter += batch_size

    # save numpy array 
    final_embeddings = np.concatenate(embeddings_list, axis=0)
    np.save("embeddings.npy", final_embeddings)
