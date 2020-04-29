import tensorflow as tf

def get_flops(keras_model_h5_path):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_model(keras_model_h5_path)

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, 
                                                  cmd='op', 
                                                  options=opts)
            return flops.total_float_ops