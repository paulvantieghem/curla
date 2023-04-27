import os
import cv2
import torch
import argparse
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import utils
from curl_sac import CurlSacAgent

def parse_args():
    parser = argparse.ArgumentParser()

    # Image source directory
    parser.add_argument('--image_dir_path', default='viz_scenes', type=str)

    # Model specifications
    parser.add_argument('--model_dir_path', default='models', type=str)
    parser.add_argument('--model_step', default=400_000, type=int)
    parser.add_argument('--encoder_type', default='pixel', type=str)

    # Image augmentation settings
    parser.add_argument('--image_height', default=76, type=int)
    parser.add_argument('--image_width', default=135, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)

    # Random seed
    parser.add_argument('--seed', default=-1, type=int)

    args = parser.parse_args()
    return args


def main():

    # Parse arguments
    args = parse_args()
    if args.seed == -1: 
        args.__dict__["seed"] = np.random.randint(1,1000000)

    # Random seed
    utils.set_seed_everywhere(args.seed)

    # Check that image directory exists and contains images
    assert os.path.exists(args.image_dir_path), 'Image directory %s does not exist' % args.image_dir_path
    assert len(os.listdir(args.image_dir_path)) > 0, 'No scenes found in %s' % args.image_dir_path

    # Image sizes
    cropped_shape = (args.image_height, args.image_width)
    obs_shape = (3*args.frame_stack, *cropped_shape)

    # Make use of GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up agent
    agent = CurlSacAgent(obs_shape=obs_shape, action_shape=(2,), device=device, encoder_type=args.encoder_type)

    # Load model
    print('Loading model %s' % os.path.join(args.model_dir_path, 'curl_%d.pt' % args.model_step))
    agent.load_curl(args.model_dir_path, str(args.model_step))

    # Run latent visualization loop
    representations = []
    scene_list = os.listdir(args.image_dir_path)
    scene_list.sort()
    for directory in scene_list:

        if directory.startswith('scene'):
            print('Visualizing latent space for %s' % directory)
            
            # Get scene frames
            frame_list = []
            dir_path = os.path.join(args.image_dir_path, directory)
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                if os.path.isfile(file_path) and file.endswith('.npy'):
                    obs = np.load(file_path)
                    frame_list.append(obs)
            assert len(frame_list) % args.frame_stack == 0
            
            # Stack frames
            obs = np.concatenate(frame_list, axis=0)

            # Get latent representation from agent
            if args.encoder_type == 'pixel':
                obs = utils.center_crop_image(obs, cropped_shape)
            with utils.eval_mode(agent):
                assert obs.shape == obs_shape, f'Shape of observations incorrect - expected {obs_shape}, got {obs.shape}'
                _ = agent.sample_action(obs)
                latent_representation = agent.actor.latent_representation
                print('Latent representation shape: %s' % str(latent_representation.shape))
                representations.append(latent_representation)
    
    # Visualize latent representations
    fig, ax = plt.subplots()
    tsne = TSNE(n_components=2)
    repr = np.concatenate(representations, axis=0)
    vec_2d = tsne.fit_transform(repr)
    ids = np.arange(1, repr.shape[0]+1)
    scatter  = ax.scatter(vec_2d[:,0], vec_2d[:,1], c=ids)
    legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Episode")
    ax.add_artist(legend)
    plt.savefig(os.path.join(args.image_dir_path, 'latent_viz.png'))

if __name__ == '__main__':
    main()