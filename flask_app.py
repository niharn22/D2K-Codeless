from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
import numpy as np
import pickle

app = Flask(__name__)

# Load the TensorFlow model
model_path = 'C:\\Users\\Nihar Nandoskar\\OneDrive\\Desktop\\D2K\\saved_model.pb'
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(model_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Load the model weights from pickle
with open('model_weights.pkl', 'rb') as f:
    model_weights = pickle.load(f)

# Define a route for object detection
@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    # Get the input image data
    image_data = request.files['image'].read()

    # Convert the image data to OpenCV format
    nparr = np.frombuffer(image_data, np.uint8)
    image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform object detection
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            # Set the loaded weights to the TensorFlow variables
            for var_name, var_value in model_weights.items():
                sess.run(tf.compat.v1.assign(var_name, var_value))

            # Define input and output tensors
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Expand dimensions
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # Run inference
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Draw bounding boxes on the image
            for i in range(len(scores[0])):
                if scores[0][i] > 0.5:  # Adjust the threshold as needed
                    ymin, xmin, ymax, xmax = boxes[0][i]
                    im_height, im_width, _ = image_np.shape
                    left, right, top, bottom = int(xmin * im_width), int(xmax * im_width), \
                                               int(ymin * im_height), int(ymax * im_height)
                    cv2.rectangle(image_np, (left, top), (right, bottom), (0, 255, 0), 3)

    # Convert the image to bytes
    retval, buffer = cv2.imencode('.jpg', image_np)
    image_bytes = buffer.tobytes()

    # Return the image with bounding boxes as bytes
    return image_bytes

if __name__ == '__main__':
    app.run(debug=True)
