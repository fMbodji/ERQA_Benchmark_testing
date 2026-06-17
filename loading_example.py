#!/usr/bin/env python3
"""
Simple example script demonstrating how to load and iterate through the ERQA dataset.
"""

import tensorflow as tf
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def parse_example(example_proto):
    """Parse a TFRecord example containing question, image, answer, and metadata."""
    feature_description = {
        'answer': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.VarLenFeature(tf.string),
        'question_type': tf.io.VarLenFeature(tf.string),
        'visual_indices': tf.io.VarLenFeature(tf.int64),
        'question': tf.io.FixedLenFeature([], tf.string)
    }

    # Parse the example
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)

    # Convert sparse tensors to dense tensors
    parsed_features['visual_indices'] = tf.sparse.to_dense(parsed_features['visual_indices'])
    parsed_features['image/encoded'] = tf.sparse.to_dense(parsed_features['image/encoded'])
    parsed_features['question_type'] = tf.sparse.to_dense(parsed_features['question_type'])

    return parsed_features

def main():
    # Path to the TFRecord file
    tfrecord_path = './data/erqa.tfrecord'
    
    # Load TFRecord dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_example)
    
    # Number of examples to display
    # num_examples = 3  - initial number
    num_examples = 15  # to get a idea of what the set of pictures look like

    
    print(f"Loading first {num_examples} examples from {tfrecord_path}...")
    print("-" * 50)
    
    # Process examples
    for i, example in enumerate(dataset.take(num_examples)):
        # Extract data from example
        answer = example['answer'].numpy().decode('utf-8')
        images_encoded = example['image/encoded'].numpy()
        question_type = example['question_type'][0].numpy().decode('utf-8') if len(example['question_type']) > 0 else "Unknown"
        visual_indices = example['visual_indices'].numpy()
        question = example['question'].numpy().decode('utf-8')
        
        print(f"\n--- Example {i+1} ---")
        print(f"Question: {question}")
        print(f"Question Type: {question_type}")
        print(f"Ground Truth Answer: {answer}")
        print(f"Number of images: {len(images_encoded)}")
        print(f"Visual indices: {visual_indices}")

       
        # Create directory for saved images
        save_dir = './data/example_images'
        os.makedirs(save_dir, exist_ok=True)

        # Display image dimensions for each image
        for j, img_encoded in enumerate(images_encoded):
            # Decode the image tensor
            img_tensor = tf.io.decode_image(img_encoded).numpy()

            # Display the raw images used in the examples
            plt.imshow(img_tensor)
            plt.title(f" Example {i+1} - Image {j+1}")
            plt.axis('off')
            #plt.show()

            # Save the image to local machine
            save_path = os.path.join(save_dir, f'example_{i+1}_image_{j+1}.png')
            plt.savefig(save_path)
            print(f" Image {j+1} saved to: {save_path}")

            print(f" Image {j+1} dimensions: {img_tensor.shape}")

        print("-" * 50)

if __name__ == "__main__":
    main() 