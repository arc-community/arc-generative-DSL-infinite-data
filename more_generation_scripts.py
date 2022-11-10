

import numpy as np





from genDSL_helpers import visualize_board








blu_shape = np.array([1,1,0, 1,0,1, 0,1,0]).reshape(3,3)
red_shape = np.array([1,0,1, 0,1,0, 1,0,1]).reshape(3,3)
grn_shape = np.array([0,1,1, 0,1,1, 1,0,0]).reshape(3,3)
mag_shape = np.array([0,1,0, 1,1,1, 0,1,0]).reshape(3,3)
shapes = [blu_shape,red_shape,grn_shape,mag_shape]
#be careful with this one, the total possible samples is 4*4=16
#so it is probably better to explicity construct the trn/val set
#using a split of all 16 possible
def generate_27a28665():
    pass
    out_board = np.zeros( (1,1), dtype=int)

    out_color_ind = int(np.random.rand()*4)
    possible_out_colors = [1,2,3,6]
    out_board[0,0] = possible_out_colors[out_color_ind]
    
    in_color_ind = int(np.random.rand()*4)
    possible_in_colors = [1,4,5,8]
    in_color = possible_in_colors[in_color_ind]
    inp_board = shapes[out_color_ind]*in_color

    # visualize_board(inp_board)
    # visualize_board(out_board)
    return inp_board, out_board

#generate_27a28665()


def generate_007bbfb7():
    #inp_board = np.zeros( (3,3), dtype=int)
    out_board = np.zeros( (9,9), dtype=int)

    inp_board = (np.random.rand(3,3)*2).astype(int)

    for y in range(3):
        for x in range(3):
            if inp_board[y,x] != 0:
                out_board[3*y:3*y+3,3*x:3*x+3] = inp_board


    color_ind = int(np.random.rand()*4)
    # changed to full color range - parapraxis
    possible_colors = [1,2,3,4,5,6,7,8,9]
    #possible_colors = [2,4,6,7]
    color = possible_colors[color_ind]

    inp_board*=color
    out_board*=color
    
    # visualize_board(inp_board)
    # visualize_board(out_board)
    return inp_board, out_board

#generate_007bbfb7()
