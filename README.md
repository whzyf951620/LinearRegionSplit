# LinearRegionSplit
Simple 2D visualization of the `transitions' in the Paper 'On the Expressive of Deep Neural Network'.
The papar employs the number of the neuron activation patterns to describe the Experssive Power of the trained Neural Networks.

高维空间中三点确定一个2D的超平面：假设A, B, C属于R^m空间且线性无关，则其三点确定的一个超平面上的任意一点X，存在X-A, B, C则是线性相关的。
即：存在不全为0的实数s, t，使得X - A = s * B + t * C。
则X = s * B + t * C + A，选取不同的实数s, t就可以生成目标超平面上的点，点足够多时，则可以生成目标平面。
