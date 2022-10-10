


import numpy as np
import matplotlib.pyplot as plt












def collect_sortOfARC_16shapes():

    sort_of_arc_shapes = np.zeros((16,3,3))

    shape_list = [[0,1,0,1,1,1,0,1,0],
                  [1,0,1,0,1,0,1,0,1,],
                  [0,0,1,0,1,0,1,0,0,],
                  [1,1,1,0,1,0,1,1,1],

                  [1,1,1,1,1,1,1,1,0],
                  [1,1,1,1,1,1,0,1,0],
                  [1,1,1,1,1,1,0,1,1],
                  [1,1,1,1,0,1,1,1,1],
                  
                  [1,0,1,1,0,1,1,1,1],
                  [1,1,1,1,0,1,1,0,1],
                  [1,0,1,0,1,1,1,1,1],
                  [1,1,1,0,1,1,1,1,1],
                  
                  [1,1,1,1,1,0,1,1,1],
                  [1,1,0,1,1,1,1,1,0],
                  [1,1,0,1,1,1,0,1,1],
                  [0,1,1,1,1,1,1,1,0],]

    shape_list_np = np.array(shape_list)
    shape_list_matr = shape_list_np.reshape((16,3,3))

    if False:
        for k in range(16):
            plt.imshow(shape_list_matr[k],cmap='magma')
            plt.show()

    if False:
        figure_3_matrix = np.zeros((9,33))
        for r in range(2):
            for c in range(8):
                print('r',r,'c',c)
                xd = shape_list_matr[r*8+c]
                y = 1+r*4
                x = 1+c*4
                figure_3_matrix[y:y+3,x:x+3] = xd
        plt.imshow(figure_3_matrix,cmap='magma')
        plt.show()

    return shape_list_matr





from genDSL_helpers import plot_riddle

sort_of_arc_shapes = collect_sortOfARC_16shapes()
#print(sort_of_arc_shapes.shape)
SHAPES = 16
COLORS = 9
#16 object shapes, 9 colors, 4 cardinal directions
INP_rule_dim = SHAPES+COLORS
OUT_rule_dim = SHAPES+COLORS+4

def generate_random_SortOfARC_boardPair(in_rule,out_rule, grid_height=20, grid_width=20):
    assert 0<=in_rule  and  in_rule<16+9
    assert 0<=out_rule and out_rule<16+9+4

    

    #original workshop paper only does 20x20
    h=grid_height
    w=grid_width
    board = np.zeros( (h,w), dtype=int)

    #original workshop paper always has exactly 3 objects
    number_of_objects = 3
    noo = number_of_objects
    #(dont change to be too large btw, because random search will not timeout)

    locs = np.zeros((noo,2),dtype=int)
    for obj in range(noo):
        
        tries = 0
        found = False
        while not found:
            y = np.random.randint(2,h-2)
            x = np.random.randint(2,w-2)
            
            if not helper_obj_far_enough(board,y,x):
                tries+=1
                #print('obj',obj,'\ttries',tries)
            else:
                found = True

                board[y,x] = 1
                locs[obj] = [y,x]
                #visualize_board(board)
                
    color_and_shapes_assigned = False
    change_indices = []
    while not color_and_shapes_assigned:
        all_colors = np.random.rand(noo)
        all_shapes = np.random.rand(noo)
        all_colors = (all_colors*COLORS).astype(int)
        all_shapes = (all_shapes*SHAPES).astype(int)

        #print(all_colors)
        #print(all_shapes)



        #this rejection technically does not ensure input=/=output
        if in_rule<SHAPES:
            if in_rule in all_shapes:
                color_and_shapes_assigned = True
                change_indices = np.argwhere(all_shapes==in_rule)[:,0]
                #print(change_indices.shape)
                #print(change_indices)
        else: #in_rule for color
            if (in_rule-SHAPES) in all_colors:
                color_and_shapes_assigned = True
                change_indices = np.argwhere(all_colors==(in_rule-SHAPES))[:,0]
                #print(change_indices.shape)
                #print(change_indices)
    
    #print(all_colors)
    #print(all_shapes)
    #visualize_board(board)
    inp_board = np.copy(board)
    out_board = np.copy(board)

    out_locs   = np.copy(locs)
    out_colors = np.copy(all_colors)
    out_shapes = np.copy(all_shapes)
    if out_rule<SHAPES:
        out_shapes[change_indices] = out_rule
    elif out_rule<SHAPES+COLORS:
        out_colors[change_indices] = (out_rule-SHAPES)
    else:
        direc = out_rule-SHAPES-COLORS
        vec = np.array( [[0,1],[-1,0],[0,-1],[1,0]][direc] )
        out_locs[change_indices] = out_locs[change_indices] + vec
        
    #print(out_colors)
    #print(out_shapes)

    for obj in range(noo):
        yx = locs[obj]
        y=yx[0];x=yx[1];
        col = all_colors[obj]+1
        shp = all_shapes[obj]
        inp_board[y-1:y+2,x-1:x+2] = sort_of_arc_shapes[shp]*col

        
        yx = out_locs[obj]
        y=yx[0];x=yx[1];
        col = out_colors[obj]+1
        shp = out_shapes[obj]
        out_board[y-1:y+2,x-1:x+2] = sort_of_arc_shapes[shp]*col
        
    #visualize_board(inp_board)
    #visualize_board(out_board)
    ##visualize_board(np.concatenate([inp_board,-1*np.ones((20,1)),out_board],axis=-1),  gridlines=True)

    return inp_board,out_board


#do they do this? it looks like it...
import matplotlib.colors as colors
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):  #https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
#magma2 = truncate_colormap(plt.get_cmap('magma'), 0.1, 0.9)
magma2 = truncate_colormap(plt.get_cmap('magma'), 0.05, 1.00)
#magma2 = truncate_colormap(plt.get_cmap('twilight'), 0.5, 1.00)
#magma2 = truncate_colormap(plt.get_cmap('magma'), 0.05, 0.85)

np.random.seed(13) #set seed for consistent dataset distribution generation
sparse_rule_matrix = np.zeros((INP_rule_dim,OUT_rule_dim))
for inp in range(INP_rule_dim):
    found=False
    while not found:
        out = (np.random.rand(1)*OUT_rule_dim).astype(int)
        if inp!=out:
            found=True
            sparse_rule_matrix[inp,out] = 1
if False:
    #plt.imshow(sparse_rule_matrix+0.05,cmap='magma') #Figure 4 in the paper
    plt.imshow(sparse_rule_matrix,cmap=magma2) 
    plt.clim(0,1)
    plt.colorbar()
    plt.show()
            
uniform_rule_matrix = np.ones((INP_rule_dim,OUT_rule_dim))
uniform_rule_matrix[np.arange(INP_rule_dim),np.arange(INP_rule_dim)] = 0
if False:
    plt.imshow(uniform_rule_matrix,cmap='magma')
    plt.colorbar()
    plt.show()
np.random.seed()

'''
all_items_same_grid_size = True  # only applies if grid_size is -1
grid_size_width = -1
grid_size_height = -1
min_grid_size = 8 # applies if grid_size is -1
'''

def gen_height_and_width(grid_height, grid_width, min_grid_size):
    if grid_height == -1:
        grid_height = np.random.randint(min_grid_size, 31)
    if grid_width == -1:
        grid_width = np.random.randint(min_grid_size, 31)
    return grid_height, grid_width

def generate_random_SortOfARC_puzzle(rule_matrix_style='sparse_rule', meta_trn_size = 5, meta_tst_size = 1, grid_width=20, grid_height=20, min_grid_size=8, all_items_same_grid_size=True, verbose=True):

    meta_trn_size = np.random.randint(2,8) if meta_trn_size==0 else meta_trn_size
    meta_tst_size = np.random.randint(2,8) if meta_tst_size==0 else meta_tst_size

    current_grid_height, current_grid_width = gen_height_and_width(grid_height, grid_width, min_grid_size)

    INP_rule_dim = SHAPES+COLORS
    OUT_rule_dim = SHAPES+COLORS+4

    #first generate a rule of the form
    #{SHAPE or COLOR} --> {SHAPE or COLOR or LOCATION}

    #in the paper, this seems to be drawn from a specific distribution
    #rather than uniform off of the diagonal
    rule_matrix = None
    if rule_matrix_style=='sparse_rule':
        rule_matrix = sparse_rule_matrix
    elif rule_matrix_style=='uniform_rule':
        rule_matrix = uniform_rule_matrix
    else:
        print("Need to define new rule matrix directly")
        quit()

    inp_rule = int(np.random.rand(1)*(INP_rule_dim))
    out_rule_noise = np.random.rand(1)*np.sum(rule_matrix[inp_rule])
    out_rule = OUT_rule_dim
    for k in range(OUT_rule_dim-1,-1,-1):
        #print(k)
        if out_rule_noise <= np.sum(rule_matrix[inp_rule,:(k+1)]):
            out_rule = k
        else:
            break
    #print("RULE",inp_rule,out_rule)



    input_grids = []
    output_grids = []
    #np.zeros((1, 2, current_grid_height, current_grid_width))
    for t in range(meta_trn_size+meta_tst_size):
        if not all_items_same_grid_size and t>0:
            current_grid_height, current_grid_width = gen_height_and_width(grid_height, grid_width, min_grid_size)
        inp,out = generate_random_SortOfARC_boardPair(inp_rule,out_rule, current_grid_height, current_grid_width)
        input_grids.append(inp)
        output_grids.append(out)

    train_input_boards = input_grids[:meta_trn_size]
    train_output_boards = output_grids[:meta_trn_size]
    test_input_boards = input_grids[meta_trn_size:]
    test_output_boards = output_grids[meta_trn_size:]

    if verbose:
        plot_riddle(train_input_boards, train_output_boards, test_input_boards, test_output_boards)
    return train_input_boards, train_output_boards, test_input_boards, test_output_boards

#make sure not to put objects on top of one another
#lest the problem become less well defined
def helper_obj_far_enough(board,y,x):
    #assumed y,x far enough from border already
    y1 = np.maximum(0,y-3);y2=np.minimum(board.shape[0],y+3+1);
    x1 = np.maximum(0,x-3);x2=np.minimum(board.shape[1],x+3+1);
    region = board[y1:y2,x1:x2]
    #print(region.shape)
    
    badness = np.sum(np.abs(region))
    if badness>0:
        return False
    else:
        return True


if __name__ == "__main__":
    for xd in range(10):
        meta_trn_boards,meta_tst_boards = generate_random_SortOfARC_puzzle()

    #generate_random_SortOfARC_boardPair(0,1)
    #generate_random_SortOfARC_boardPair(0,27)
    #generate_random_SortOfARC_boardPair(20,27)






















