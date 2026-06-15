import tensorflow as tf

examples = [
    {
        image_path : 'data/custom_images/.....jpeg',
        number_of_images : 1,
        question : "If the robot gripper picks up the - shaped object, where will the robot place it? Choices: A. Robot places it on one of the - holes. B. Robot places it on one of the triangular holes. C. Robot places it on one of the circular holes. D. Robot places it on one of the circular holes. Please answer directly with only the letter of the correct option and nothing else.",
        question_type : "Action Reasoning",
        answer : 'A',
        visual_indices : []
    },
    {
        image_path : '',
        question : "",
        question_type : "Trajectory Reasoning",
        answer : '',
        visual_indices : []
    },
]