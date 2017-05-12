import collections
import numpy as np
import tensorflow as tf
import math
import time

#-------------------------------数据预处理---------------------------#  

poetry_file = 'poetry.txt'

# 诗集
poetrys = []
with open(poetry_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            title, content = line.strip().split(':')
            content = content.replace(' ', '')
            if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
                continue
            if len(content) < 5 or len(content) > 79:
                continue
            content = '[' + content + ']'
            poetrys.append(content)
        except Exception as e:
            pass

# 按诗的字数排序  
poetrys = sorted(poetrys, key=lambda line: len(line))
print('诗集长度：', len(poetrys))

# 统计每个字出现次数  
all_words = []
for poetry in poetrys:
    all_words += [word for word in poetry]
counter = collections.Counter(all_words)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
words, _ = zip(*count_pairs)

# 取前多少个常用字  
words = words[:len(words)] + (' ',)

# 每个字映射为一个数字ID  
word_num_map = dict(zip(words, range(len(words))))

# 把诗转换为向量形式
to_num = lambda word: word_num_map.get(word, len(words))
poetrys_vector = [list(map(to_num, poetry)) for poetry in poetrys]

batch_size = 1

n_chunk = len(poetrys_vector)

x_batches = []
y_batches = []

for i in range(math.ceil(n_chunk / batch_size)):
    start_index = i * batch_size
    end_index = start_index + batch_size

    batches = poetrys_vector[start_index:end_index]

    length = max(map(len, batches))
    xdata = np.full((batch_size, length), word_num_map[' '], np.int32)
    for row in range(len(batches)):
        xdata[row, :len(batches[row])] = batches[row]
    ydata = np.copy(xdata)
    ydata[:, :-1] = xdata[:, 1:]
    x_batches.append(xdata)
    y_batches.append(ydata)

#---------------------------------------RNN--------------------------------------#
input_data = tf.placeholder(tf.int32, [batch_size, None])
output_targets = tf.placeholder(tf.int32, [batch_size, None])

# 定义RNN
def neural_network(model='lstm', rnn_size=128, num_layers=2):
    def build_rnn_cell():
        if model == 'rnn':
            cell_fun = tf.contrib.rnn.BasicRNNCell
        elif model == 'gru':  
            cell_fun = tf.contrib.rnn.GRUCell  
        elif model == 'lstm':  
            cell_fun = tf.contrib.rnn.BasicLSTMCell
        
        cell = cell_fun(rnn_size, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        return cell
    
    multicell = tf.contrib.rnn.MultiRNNCell([build_rnn_cell() for _ in range(num_layers)], state_is_tuple=True)

    initial_state = multicell.zero_state(batch_size, tf.float32)

    with tf.variable_scope('rnnlm'):
        softmax_w = tf.get_variable("softmax_w", [rnn_size, len(words) + 1])
        softmax_b = tf.get_variable("softmax_b", [len(words) + 1])
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [len(words) + 1, rnn_size])
            inputs = tf.nn.embedding_lookup(embedding, input_data)
    
    outputs, last_state = tf.nn.dynamic_rnn(multicell, inputs, initial_state=initial_state, scope='rnnlm')
    output = tf.reshape(outputs, [-1, rnn_size])

    logits = tf.matmul(output, softmax_w) + softmax_b
    probs = tf.nn.softmax(logits)
    return logits, last_state, probs, multicell, initial_state

#-------------------------------生成古诗---------------------------------#  
# 使用训练完成的模型  

def gen_poetry_with_head(head):  
    def to_word(weights):  
        t = np.cumsum(weights)  
        s = np.sum(weights)  
        sample = int(np.searchsorted(t, np.random.rand(1)*s))
        return words[sample]  
   
    _, last_state, probs, cell, initial_state = neural_network()  
   
    with tf.Session() as sess:  
        sess.run(tf.global_variables_initializer())  
   
        saver = tf.train.Saver(tf.global_variables())  
        saver.restore(sess, 'poetry.module-105')  
   
        state_ = sess.run(cell.zero_state(1, tf.float32))  
        poem = ''  
        i = 0  
        for word in head:  
            while word != '，' and word != '。' and word != ' ':  
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

print(gen_poetry_with_head('边边最好了哈'))
    