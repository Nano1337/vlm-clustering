import h5py
import matplotlib.pyplot as plt
import argparse

file_path = "/home/haoli/Documents/data/nsd_images/nsd_stimuli.hdf5"

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=None, help="Image ID to visualize") 
    args = parser.parse_args()

    if args.id is None: 
        raise NotImplementedError("Use --id flag to specify image")

    with h5py.File(file_path, 'r') as f:
        data = f['imgBrick']

        image = data[args.id]
        print(image.shape)
        # Display the image
        plt.imshow(image)  # 'cmap' specifies the color map
        plt.axis('off')  # Optional: Removes the axis
        plt.savefig("figs/image.png")