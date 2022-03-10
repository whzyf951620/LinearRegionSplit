import os

for num in [1000000*50]: #, 50000*50, 100000*50, 200000*50, 500000*50, 700000*50, 1000000*50]:
    # os.system('python ellipse_CountTtansition_MNIST.py -n ' + str(num) + ' -sn 50')
    os.system('python ellipse_CountTtansition_MNIST.py -n ' + str(num) + ' -sn 50' + ' >info' + str(num) + '.txt 2>&1')
