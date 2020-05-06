def init_keras():
    print("Applying CUDNN runtime fix...")
    from tensorflow.compat.v1.keras.backend import set_session
    import tensorflow.compat.v1 as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)
 
