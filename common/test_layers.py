import numpy as np

import layers

def main():
    input_X = [[1,2,3],[4,5,6],[7,8,9]]
    W = [[1,1,1],[0,0,0],[1,1,1]]
    b = [0]
    
    input_X = np.array(input_X).reshape((1,3,3,1))
    W = np.array(W).reshape((1,3,3,1))
    b = np.array(b).reshape((1,1,1,1))

    conv = layers.Conv(num_input_channels=1,filter_size=3,padding='valid',stride=2,
            num_filters=3,initializer='xavier_normal')
    
    
    output_Z,cache = conv.forward(input_X)
    print("### test convolutional layer ###")
    print("input_X = " ,input_X)
    print("W = ",cache[1])
    print("output_Z = " ,output_Z.shape)

if __name__ == "__main__":
    main()
