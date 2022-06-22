import tensorflow as tf
import numpy as np

def naive_model(X, y, num_classes=250):
    c1 = tf.layers.conv2d(X, 32, [7, 7], padding='SAME') # 128 x 128 x 32
    b1 = tf.layers.batch_normalization(c1)
    h1 = tf.nn.relu(b1)
    p1 = tf.layers.max_pooling2d(h1, [2, 2], [2, 2]) # 64 x 64 x 32
    
    c2 = tf.layers.conv2d(p1, 64, [5, 5], padding='SAME') # 64 x 64 x 64
    b2 = tf.layers.batch_normalization(c2)
    h2 = tf.nn.relu(b2)
    p2 = tf.layers.max_pooling2d(h2, [2, 2], [2, 2]) # 32 x 32 x 64
    
    c3 = tf.layers.conv2d(p2, 128, [3, 3], padding = 'SAME') # 32 x 32 x 128
    b3 = tf.layers.batch_normalization(c3)
    h3 = tf.nn.relu(b3)
    p3 = tf.layers.max_pooling2d(h3, [2, 2], [2, 2]) # 16 x 16 x 128
    
    #p4 = tf.layers.average_pooling2d(p3, [32, 32], [1, 1]) # 1 x 1 x 64
    
    p3_flat = tf.reshape(p3, [-1,32768])
    y_out = tf.layers.dense(p3_flat, num_classes)
    
    return y_out


def resnet(X, y, layer_depth=4, num_classes=250, reg=1e-2, is_training=True):
    # RESnet-ish
    l2_reg = tf.contrib.layers.l2_regularizer(reg)

    """
    Input: 128x128x1
    Output: 64x64x64
    """
    c0 = tf.layers.conv2d(X, 64, [7, 7], strides=[2, 2], padding='SAME', kernel_regularizer=l2_reg)
    c0 = tf.layers.batch_normalization(c0, training=is_training)
    match_dimensions = True
    for i in range(layer_depth):
        c1 = tf.layers.conv2d(c0, 64, [3, 3], padding='SAME', kernel_regularizer=l2_reg) #conv
        b1 = tf.layers.batch_normalization(c1, training=is_training) #bn
        h1 = tf.nn.relu(b1) #relu
        c2 = tf.layers.conv2d(h1, 64, [3, 3], padding='SAME', kernel_regularizer=l2_reg) #conv
        b2 = tf.layers.batch_normalization(c2, training=is_training) #bn
        r = c0 + b2
        c0 = tf.nn.relu(r)
    
    """
    Input: 64x64x64
    Output: 32x32x128
    """
    downsample = True
    for i in range(layer_depth):
        c1 = tf.layers.conv2d(c0, 128, [3, 3], 
                              strides=([2, 2] if downsample else [1, 1]),
                              padding='SAME',
                              kernel_regularizer=l2_reg)
        b1 = tf.layers.batch_normalization(c1, training=is_training) #bn
        h1 = tf.nn.relu(b1) #relu
        c2 = tf.layers.conv2d(h1, 128, [3, 3], padding='SAME', kernel_regularizer=l2_reg) #conv
        b2 = tf.layers.batch_normalization(c2, training=is_training) #bn
        if downsample:
            c0_proj = tf.layers.conv2d(c0, 128, [1, 1], padding='SAME', kernel_regularizer=l2_reg)
            c0_proj = tf.layers.average_pooling2d(c0_proj, (2, 2), (2, 2))
            r = c0_proj + b2
            downsample = False
        else:
            r = c0 + b2
        c0 = tf.nn.relu(r)

    """
    Input: 32x32x128
    Output: 16x16x256
    """
    downsample = True
    for i in range(layer_depth):
        c1 = tf.layers.conv2d(c0, 256, [3, 3], 
                              strides=([2, 2] if downsample else [1, 1]),
                              padding='SAME',
                              kernel_regularizer=l2_reg)
        b1 = tf.layers.batch_normalization(c1, training=is_training) #bn
        h1 = tf.nn.relu(b1) #relu
        c2 = tf.layers.conv2d(h1, 256, [3, 3], padding='SAME', kernel_regularizer=l2_reg) #conv
        b2 = tf.layers.batch_normalization(c2, training=is_training) #bn
        if downsample:
            c0_proj = tf.layers.conv2d(c0, 256, [1, 1], padding='SAME', kernel_regularizer=l2_reg)
            c0_proj = tf.layers.average_pooling2d(c0_proj, (2, 2), (2, 2))
            r = c0_proj + b2
            downsample = False
        else:
            r = c0 + b2
        c0 = tf.nn.relu(r)

    """
    Input: 16x16x256
    Output: 8x8x512
    """
    downsample = True
    for i in range(layer_depth):
        c1 = tf.layers.conv2d(c0, 512, [3, 3], 
                              strides=([2, 2] if downsample else [1, 1]),
                              padding='SAME',
                              kernel_regularizer=l2_reg)
        b1 = tf.layers.batch_normalization(c1, training=is_training) #bn
        h1 = tf.nn.relu(b1) #relu
        c2 = tf.layers.conv2d(h1, 512, [3, 3], padding='SAME', kernel_regularizer=l2_reg) #conv
        b2 = tf.layers.batch_normalization(c2, training=is_training) #bn
        if downsample:
            c0_proj = tf.layers.conv2d(c0, 512, [1, 1], padding='SAME', kernel_regularizer=l2_reg)
            c0_proj = tf.layers.average_pooling2d(c0_proj, (2, 2), (2, 2))
            r = c0_proj + b2
            downsample = False
        else:
            r = c0 + b2
        c0 = tf.nn.relu(r)
    
    p1 = tf.layers.average_pooling2d(c0, (8, 8), (1,1))
    p1_flat = tf.reshape(p1, [-1, 512])
    y_out = tf.layers.dense(p1_flat, num_classes, kernel_regularizer=l2_reg)
    
    return y_out

def resnet_dropout(shape = (128, 128, 1), layer_depth=4, num_classes=250, reg=1e-2):
    
    X_input = Input(shape)
    l2_reg = L2(reg)

    """
    Input: 128x128x1
    Output: 64x64x64
    """
    d0 = Dropout(rate=0.2)(X_input)
    c0 = Conv2D(64, [7, 7], strides=[2, 2], padding='SAME', kernel_regularizer=l2_reg)(d0)
    c0 = BatchNormalization()(c0)
    match_dimensions = True
    for i in range(layer_depth):
        c1 = Conv2D(64, [3, 3], padding='SAME', kernel_regularizer=l2_reg)(c0) #conv
        b1 = BatchNormalization()(c1) #bn
        h1 = Activation('relu')(b1) #relu
        c2 = Conv2D(64, [3, 3], padding='SAME', kernel_regularizer=l2_reg)(h1) #conv
        b2 = BatchNormalization()(c2) #bn
        r = c0 + b2
        c0 = Activation('relu')(r)
    
    """
    Input: 64x64x64
    Output: 32x32x128
    """
    downsample = True
    for i in range(layer_depth):
        c1 = Conv2D(128, [3, 3], 
                              strides=([2, 2] if downsample else [1, 1]),
                              padding='SAME',
                              kernel_regularizer=l2_reg)(c0)
        b1 = BatchNormalization()(c1) #bn
        h1 = Activation('relu')(b1) #relu
        c2 = Conv2D(128, [3, 3], padding='SAME', kernel_regularizer=l2_reg)(h1) #conv
        b2 = BatchNormalization()(c2) #bn
        if downsample:
            c0_proj = Conv2D(128, [1, 1], padding='SAME', kernel_regularizer=l2_reg)(c0)
            c0_proj = AveragePooling2D((2, 2), (2, 2))(c0_proj)
            r = c0_proj + b2
            downsample = False
        else:
            r = c0 + b2
        c0 = Activation('relu')(r)

    """
    Input: 32x32x128
    Output: 16x16x256
    """
    downsample = True
    for i in range(layer_depth):
        c1 = Conv2D(256, [3, 3], 
                              strides=([2, 2] if downsample else [1, 1]),
                              padding='SAME',
                              kernel_regularizer=l2_reg)(c0)
        b1 = BatchNormalization()(c1) #bn
        h1 = Activation('relu')(b1) #relu
        c2 = Conv2D(256, [3, 3], padding='SAME', kernel_regularizer=l2_reg)(h1) #conv
        b2 = BatchNormalization()(c2) #bn
        if downsample:
            c0_proj = Conv2D(256, [1, 1], padding='SAME', kernel_regularizer=l2_reg)(c0)
            c0_proj = AveragePooling2D((2, 2), (2, 2))(c0_proj)
            r = c0_proj + b2
            downsample = False
        else:
            r = c0 + b2
        c0 = Activation('relu')(r)

    """
    Input: 16x16x256
    Output: 8x8x512
    """
    downsample = True
    for i in range(layer_depth):
        c1 = Conv2D(512, [3, 3], 
                              strides=([2, 2] if downsample else [1, 1]),
                              padding='SAME',
                              kernel_regularizer=l2_reg)(c0)
        b1 = BatchNormalization()(c1) #bn
        h1 = Activation('relu')(b1) #relu
        c2 = Conv2D(512, [3, 3], padding='SAME', kernel_regularizer=l2_reg)(h1) #conv
        b2 = BatchNormalization()(c2) #bn
        if downsample:
            c0_proj = Conv2D(512, [1, 1], padding='SAME', kernel_regularizer=l2_reg)(c0)
            c0_proj = AveragePooling2D((2, 2), (2, 2))(c0_proj)
            r = c0_proj + b2
            downsample = False
        else:
            r = c0 + b2
        c0 = Activation('relu')(r)
    
    p1 = AveragePooling2D((8, 8), (1,1))(c0)
    p1_flat = tf.reshape(p1, (-1, 512))
    d1 = Dropout(rate=0.2)(p1_flat)
    y_out = Dense(num_classes, kernel_regularizer=l2_reg)(d1)
    y_out = Activation('softmax')(y_out)
    
    model = keras.models.Model(inputs = X_input, outputs = y_out, name = 'ResNetDropoutBozaLuka')

    return model


