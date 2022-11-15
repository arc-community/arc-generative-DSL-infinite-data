

import numpy as np

import random



from genDSL_helpers import visualize_board








blu_shape = np.array([1,1,0, 1,0,1, 0,1,0]).reshape(3,3)
red_shape = np.array([1,0,1, 0,1,0, 1,0,1]).reshape(3,3)
grn_shape = np.array([0,1,1, 0,1,1, 1,0,0]).reshape(3,3)
mag_shape = np.array([0,1,0, 1,1,1, 0,1,0]).reshape(3,3)
shapes = [blu_shape,red_shape,grn_shape,mag_shape]
#be careful with this one, the total possible samples is 4*4=16
#so it is probably better to explicity construct the trn/val set
#using a split of all 16 possible
def generate_27a28665(color_set, out_color_ind):
    pass
    out_board = np.zeros( (1,1), dtype=int)

    #out_color_ind = int(np.random.rand()*4)

    possible_out_colors = []
    # add permutations of 4 colors between 1 and 9
    for i in range(1,10):
        for j in range(1,10):
            for k in range(1,10):
                for l in range(1,10):
                    possible_out_colors.append([i,j,k,l])
    # how many combinations is that? 9*9*9*9 = 6561

    possible_out_colors = possible_out_colors[color_set]
    out_board[0,0] = possible_out_colors[out_color_ind]
    
    #in_color_ind = int(np.random.rand()*4)
    possible_in_colors = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    in_color = random.choice(possible_in_colors)
    inp_board = shapes[out_color_ind]*in_color

    # visualize_board(inp_board)
    # visualize_board(out_board)
    return inp_board, out_board

def generate_full_riddle_27a28665(n_train_items=5):
    '''
    Generate n_train_items input/output pairs for the 27a28665 riddle
    Generate 1 test item that has out_color_ind that is used in the training set
    '''
    train_inp_boards = []
    train_out_boards = []
    train_out_color_inds = []
    # random number between 0 and 6560
    color_set = random.randint(0,6560)
    for i in range(n_train_items):
        out_color_ind = int(np.random.rand()*4)
        inp_board, out_board = generate_27a28665(color_set, out_color_ind)
        train_inp_boards.append(inp_board)
        train_out_boards.append(out_board)
        train_out_color_inds.append(out_color_ind)
    test_inp_board, test_out_board = generate_27a28665(color_set, out_color_ind)
    return train_inp_boards, train_out_boards, test_inp_board, test_out_board

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
