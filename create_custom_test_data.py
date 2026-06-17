import tensorflow as tf
import os

examples = [
    {
        'image_paths': ['data/custom_test_images/q1-image.jpg'],
        'question': "If the robot gripper picks up the L shaped metal object, what color arrow should the robot follow to place it in the right spot? Choices: A. Blue arrow. B. Yellow arrow. C. Red arrow. D. Black arrow. Please answer directly with only the letter of the correct option and nothing else.",
        'question_type': "Trajectory Reasoning",
        'answer': 'A',
        'visual_indices': [0],
    },
    {
        'image_paths': ['data/custom_test_images/q2-image.jpg'],
        'question': "The red dot indicate where the green object under the robot gripper should be placed, what action could the robot perform to place it? Choices: A. Close gripper from its current position and move up. B. Move down, grab the object, move lower and release. C. Move down, grab the object, move right and release. D. No action needed. Please answer directly with only the letter of the correct option and nothing else.",
        'question_type': "Action Reasoning",
        'answer': 'C',
        'visual_indices': [0],
    },
    {
        'image_paths': ['data/custom_test_images/q3-image.jpg'],
        'question': "There are four points marked with letters, which one is closer to the robot gripper, so that the picked object can be placed in its correct fitted place? Choices: A. A. B. B. C. C. D. D. Please answer directly with only the letter of the correct option and nothing else.",
        'question_type': "Pointing",
        'answer': 'B',
        'visual_indices': [0],
    },
]

def encode_images(image_paths):
    encoded = []
    for path in image_paths:
        with open(path, 'rb') as f:
            encoded.append(f.read())
    return encoded


def make_tfrecord_example(ex):
    encoded_images = encode_images(ex['image_paths'])

    feature = {
        'answer': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ex['answer'].encode('utf-8')])),
        'question': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ex['question'].encode('utf-8')])),
        'question_type': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ex['question_type'].encode('utf-8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded_images)),
        'visual_indices': tf.train.Feature(int64_list=tf.train.Int64List(value=ex['visual_indices'])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def main():
    output_path = 'data/custom_test.tfrecord'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with tf.io.TFRecordWriter(output_path) as writer:
        for i, ex in enumerate(examples):
            tf_example = make_tfrecord_example(ex)
            writer.write(tf_example.SerializeToString())
            print(f"Written example {i+1}: {ex['question_type']}")

    print(f"\nSaved {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    main()
