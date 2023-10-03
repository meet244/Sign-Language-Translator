import numpy as np
import tensorflow as tf

class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/model.tflite',
        num_threads=1,
    ):
        # Initialize the KeyPointClassifier object
        # Parameters:
        # - model_path: Path to the TensorFlow Lite model file
        # - num_threads: Number of threads to use for model inference (default is 1)

        # Create an interpreter for the TensorFlow Lite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        # Allocate memory for the interpreter
        self.interpreter.allocate_tensors()

        # Get input and output details of the model
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        # Perform inference using the KeyPointClassifier
        # Parameters:
        # - landmark_list: A list of landmarks to classify
        
        # Get the index of the input tensor
        input_details_tensor_index = self.input_details[0]['index']
        
        # Set the input tensor with the landmark_list data
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))

        # Run inference
        self.interpreter.invoke()

        # Get the index of the output tensor
        output_details_tensor_index = self.output_details[0]['index']

        # Get the result from the output tensor
        result = self.interpreter.get_tensor(output_details_tensor_index)

        # Find the index with the highest confidence score
        result_index = np.argmax(np.squeeze(result))

        # Return the index of the predicted class
        return result_index
