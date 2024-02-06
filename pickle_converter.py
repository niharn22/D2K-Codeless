import tensorflow as tf
import pickle

# Load the TensorFlow model
model_path = 'path/to/your/model.pb'
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(model_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Get the model weights
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        model_weights = {}
        for var in tf.compat.v1.global_variables():
            model_weights[var.name] = sess.run(var)

# Save the model weights as a pickle file
with open('model_weights.pkl', 'wb') as f:
    pickle.dump(model_weights, f)
