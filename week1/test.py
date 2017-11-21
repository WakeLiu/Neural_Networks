import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

node1 = tf.constant(3) # 常數3的tensor
node2 = tf.constant(4) # 常數4的tensor
print(node1, node2)

sess = tf.Session() # 創建一個tensorflow session
print("sess.run(node1),sess.run(node2): ",sess.run([node1, node2])) # 把node1 node2丟進session中run

node3 = tf.add(node1, node2) # 把前面創建的const tensor node1、node2相加
print("node3: ", node3) # node3就是相加後的結果
print("sess.run(node3): ",sess.run(node3))

"""
Placeholder

正如上面所提到，在Tensorflow中我們都是先建好Graph再決定資料的input與output，
這時候我們就需要Placeholder來幫助我們在還沒有資料的時候先佔個位子(正如其名)。
"""
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # 這行等效於 adder_node = tf.add(a, b)

"""
接著我們當然又是找session幫我們執行這個Graph，
可以看到在sess.run的參數中我們除了第一個參數指定了這次要run的Output外，
在第二個參數我們給了一個dictionary，這就是我們這次run的過程賦予a和b兩個Placeholder的值，
因此adder_node的計算就會根據我們feed進去的資料作改變，在這邊很顯然的答案就是3+4.5=7.5。
"""

print(sess.run(adder_node, {a: 3, b:4.5}))

"""


基本單位叫做Tensor(高維度矩陣)，我們其實可以餵給placeholder任何維度任何長度的data。


"""
print(sess.run(adder_node, {a: [1, 3, 5], b: [2, 4, 6]}))
