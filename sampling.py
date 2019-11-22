def tail_free(logits, z, temperature=1.0):
    """
    Inputs:
    * logits (tensorflow tensor, Batch size x number of tokens) - takes in the neural network output
    * z (float) - hyperparameter for where to draw the tail. Recommend a value of 0.9 - 0.95. The lower
    the value the fewer tokens are kept (tighter the tail is).
    * temperature (float) - optional temperature parameter. 
    
    Outputs: 
    * samples - tensor (Batch size x 1) - randomly sampled tokens post pruning
    """
    logits = logits / tf.to_float(temperature)
    sps = tf.sort(tf.nn.softmax(logits, axis=1), direction='DESCENDING',axis=1)
    grad = sps[:,1:]-sps[:,:-1] # first derivative
    grad = grad[:,1:]-grad[:,:-1] #this is the 2nd derivative

    only_pos = tf.math.abs(grad)
    sec_indices = tf.range(grad.shape[1].value)
    sec_weights = only_pos/ tf.math.reduce_sum( only_pos, axis=1, keepdims=True )
    
    tail_ids = tf.cast(tf.argmax(tf.cast(tf.cumsum(sec_weights, axis=1)>z, tf.int8), axis=1), tf.int32)+1 
    # adding one to put it in the center of the tail.

    logit_inds = tf.stack([tf.range(0,logits.shape[0].value), tail_ids], axis=1)
    tail_min_vals = tf.expand_dims(tf.gather_nd(logits, logit_inds),1)

    # removes any tokens below the tail location by setting their values to be very very small.
    pruned = tf.where(
            logits < tail_min_vals,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    # do not need to convert to softmax again (renormalize) before passing to tf.multinomial
    samples = tf.multinomial(pruned, num_samples=1, output_dtype=tf.int32)
    return samples
