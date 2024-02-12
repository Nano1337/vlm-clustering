import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def append_batch_to_parquet(df, parquet_file): 

    table = pa.Table.from_pandas(df)
    schema = table.schema
    writer = pq.ParquetWriter(parquet_file, schema, compression='snappy')
    writer.write_table(table)

    if writer: 
        writer.close()
        print("Successfully added to parquet file")

if __name__ == "__main__": 

    parquet_file = "data.parquet"

    data = {
        '__key__': [
            "sample0001", # horse riding 
            "sample0002", # water well spout on street
            "sample0003", # horse jumping
            ],
        'caption': [
            "In the tranquil setting of a serene lake, two individuals are seen enjoying a horseback ride. The person on the left, donned in a blue shirt, is astride a majestic brown horse. The horse, with its coat gleaming under the sunlight, stands still, perhaps taking in the beauty of the surroundings. On the right, another person is seen riding a white horse. The horse, with its coat as pure as snow, stands calmly, mirroring the peacefulness of the scene. The lake, reflecting the clear blue sky above, stretches out in front of them. It's a beautiful day with the sun shining brightly, casting a warm glow on the scene. In the background, a line of trees stands tall, their green leaves rustling gently in the breeze. The trees add a touch of nature's vibrancy to this picturesque setting. Overall, this image captures a moment of peace and tranquility, as these two individuals enjoy their horseback ride by the lake.", 
            "The image captures a quaint scene on a city street. Dominating the foreground is a black fire hydrant, standing tall and sturdy. It's not just any hydrant, but one with a unique design - a curved spout on the left side and a straight spout on the right, both ready to serve their purpose. The hydrant is strategically placed on a roundabout, a circular area paved with gray bricks that add a touch of urban charm to the scene. The bricks are neatly arranged in a circular pattern, creating a sense of order amidst the hustle and bustle of the city. In the background, life goes on as usual. People can be seen walking on the sidewalk, going about their day. Buildings line the street, their windows reflecting the world around them. A solitary tree stands tall, its leaves rustling in the breeze. The image is taken from a low angle, making the hydrant appear even more imposing. The perspective also allows for a glimpse of the sky, adding depth to the scene. Despite being an inanimate object, the hydrant seems to be an integral part of this urban landscape, standing as a silent sentinel on the city street.",
            "The image shows a person riding a brown horse, captured in mid-jump over an obstacle. The rider is wearing a brown jacket, white pants, and a black helmet, which suggests they are participating in an equestrian event, likely show jumping. The horse is in full gallop, with its front legs extended forward and its hind legs bent at the knees, indicating the height of the jump. The rider's body is leaning forward, with their hands holding the reins and their legs firmly in the stirrups, demonstrating the athleticism and skill required for this sport. The background is a cloudy sky, which adds a dramatic effect to the image. The focus is on the horse and rider, with the background being out of focus, emphasizing the action and movement of the scene.",
        ]
    }

    # load dictionary data into pandas dataframe
    df = pd.DataFrame(data)
    print(df)

    append_batch_to_parquet(df, parquet_file)
