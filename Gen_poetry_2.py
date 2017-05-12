def gen_poetry_with_head(head):  
    def to_word(weights):  
        t = np.cumsum(weights)  
        s = np.sum(weights)  
        sample = int(np.searchsorted(t, np.random.rand(1)*s))  
        return words[sample]  
   
    _, last_state, probs, cell, initial_state = neural_network()  
   
    with tf.Session() as sess:  
        sess.run(tf.initialize_all_variables())  
   
        saver = tf.train.Saver(tf.all_variables())  
        saver.restore(sess, 'poetry.module-49')  
   
        state_ = sess.run(cell.zero_state(1, tf.float32))  
        poem = ''  
        i = 0  
        for word in head:  
            while word != '，' and word != '。':  
                poem += word  
                x = np.array([list(map(word_num_map.get, word))])  
                [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})  
                word = to_word(probs_)  
                time.sleep(1)  
            if i % 2 == 0:  
                poem += '，'  
            else:  
                poem += '。'  
            i += 1  
        return poem  
   
print(gen_poetry_with_head('一二三四'))