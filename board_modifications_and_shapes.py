from logging import root
import numpy as np
import random
from genDSL_helpers import *

def create_blank_board(height, width, bg_color):
    # create a new numpy array of the specified size with the background color
    board_np = np.zeros((height, width), dtype=np.int8)
    board_np.fill(bg_color)
    return board_np

def get_board_bg_color(board_np) -> int:
    """
    Returns the background color of a board (the number that appears the most)
    """
    return np.bincount(board_np.flatten()).argmax()

def resize_array_padding_left_and_top(array_to_reshape, array_to_copy_size_from, pad_with=0):
    """
    This only works if the array_to_reshape is smaller than array_to_copy_size_from.
    Resize numpy ndarray array_to_reshape to be the same size as array_to_copy_size_from.
    The new cells should be pad_with and added to the left and top of ndarray array_to_reshape.
    """
    # get the dimensions of array_to_reshape
    dimensions = array_to_reshape.shape

    # get the dimensions of array_to_copy_size_from
    compare_dimensions = array_to_copy_size_from.shape

    # array to hold the new larger array
    new_array = np.zeros(compare_dimensions, dtype=array_to_reshape.dtype)
    new_array.fill(pad_with)
    
    start_x = compare_dimensions[1] - dimensions[1]
    start_y = compare_dimensions[0] - dimensions[0]

    # copy the smaller array into the larger array starting at x,y coordinates
    # only resize if the array_to_reshape is smaller than array_to_copy_size_from
    if dimensions[0] > compare_dimensions[0] or dimensions[1] > compare_dimensions[1]:
        if dimensions[0] > compare_dimensions[0] and dimensions[1] > compare_dimensions[1]:
            new_array[0:compare_dimensions[0], 0:compare_dimensions[1]] = array_to_reshape[0:compare_dimensions[0], 0:compare_dimensions[1]]
        elif dimensions[0] > compare_dimensions[0]:
            new_array[0:compare_dimensions[0], start_x:start_x+dimensions[1]] = array_to_reshape[0:compare_dimensions[0], 0:dimensions[1]]
        elif dimensions[1] > compare_dimensions[1]:
            new_array[start_y:start_y+dimensions[0], 0:compare_dimensions[1]] = array_to_reshape[0:dimensions[0], 0:compare_dimensions[1]]
    else: 
        new_array[start_y:start_y+dimensions[0],start_x:start_x+dimensions[1]] = array_to_reshape
        #new_array[0:dimensions[0],0:dimensions[1]] = array_to_reshape

    # return the new larger array
    return new_array

# draw t minus
def draw_symetric_t_shape(fg, scale=1, bg=0):
    """
    Draws an upside down T (tetris style)
    """
    # create a blank board
    board_np = create_blank_board(scale*2, scale*3, bg)
    # draw the shape on the board
    board_np[0:scale, scale:scale*2] = fg
    board_np[scale:scale*2, 0:scale*3] = fg
    # return the board
    return board_np

def draw_t_shape(fg, scale=1, bg=0):
    """
    Draws a T
    """
    # create a blank board
    board_np = create_blank_board(scale*3, scale*3, bg)
    # draw the shape on the board
    board_np[0:scale, 0:scale*3] = fg
    board_np[scale:scale*3, scale:scale*2] = fg
    # return the board
    return board_np

def draw_c_shape(fg, scale=1, bg=0):
    """
    Draws a C
    """
    # create a blank board
    board_np = create_blank_board(scale*3, scale*3, bg)
    # draw the shape on the board
    board_np[0:scale*3, 0:scale] = fg
    board_np[0:scale, 0:scale*3] = fg
    board_np[scale*2:scale*3, 0:scale*3] = fg
    # return the board
    return board_np

def draw_c_shallow_shape(fg, scale=1, bg=0):
    """
    Draws a C
    """
    # create a blank board
    board_np = create_blank_board(scale*3, scale*3, bg)
    # draw the shape on the board
    board_np[0:scale*3, 0:scale] = fg
    board_np[0:scale, 0:scale*2] = fg
    board_np[scale*2:scale*3, 0:scale*2] = fg

    # return the board
    return board_np

def draw_solid_square_shape(fg, scale=1, bg=0):
    """
    Draws a solid square
    """
    # create a blank board
    board_np = create_blank_board(scale, scale, bg)
    # draw the shape on the board
    board_np[0:scale, 0:scale] = fg
    # return the board
    return board_np

def draw_cross_shape(fg, scale=1, bg=0):
    """
    Draws a cross
    """
    # create a blank board
    board_np = create_blank_board(scale*3, scale*3, bg)
    # draw the shape on the board
    board_np[scale:scale*2, 0:scale*3] = fg
    board_np[0:scale*3, scale:scale*2] = fg
    # return the board
    return board_np

def draw_left_z_shape(fg, scale=1, bg=0):
    """
    Draws a left z
    """
    # create a blank board
    board_np = create_blank_board(scale*2, scale*3, bg)
    # draw the shape on the board
    board_np[0:scale, 0:scale*2] = fg
    board_np[scale:scale*2, scale:scale*3] = fg
    # return the board
    return board_np

def draw_right_z_shape(fg, scale=1, bg=0):
    """
    Draws a right z
    """
    # create a blank board
    board_np = create_blank_board(scale*2, scale*3, bg)
    # draw the shape on the board
    board_np[0:scale, scale:scale*3] = fg
    board_np[scale:scale*2, 0:scale*2] = fg
    # return the board
    return board_np

def draw_left_L_shape(fg, scale=1, bg=0):
    """
    Draws a left L
    """
    # create a blank board
    board_np = create_blank_board(scale*3, scale*2, bg)
    # draw the shape on the board
    board_np[0:scale*3, 0:scale] = fg
    board_np[0:scale, scale:scale*2] = fg
    # return the board
    return board_np

def draw_right_L_shape(fg, scale=1, bg=0):
    """
    Draws a right L
    """
    # create a blank board
    board_np = create_blank_board(scale*3, scale*2, bg)
    # draw the shape on the board
    board_np[0:scale*3, scale:scale*2] = fg
    board_np[0:scale, 0:scale] = fg
    # return the board
    return board_np

def draw_left_s_shape(fg, scale=1, bg=0):
    """
    Draws a left s
    """
    # create a blank board
    board_np = create_blank_board(scale*2, scale*3, bg)
    # draw the shape on the board
    board_np[0:scale, scale:scale*2] = fg
    board_np[scale:scale*2, 0:scale] = fg
    # return the board
    return board_np

def draw_right_s_shape(fg, scale=1, bg=0):
    """
    Draws a right s
    """
    # create a blank board
    board_np = create_blank_board(scale*2, scale*3, bg)
    # draw the shape on the board
    board_np[0:scale, 0:scale] = fg
    board_np[scale:scale*2, scale:scale*2] = fg
    # return the board
    return board_np

def draw_elbow_shape(fg, scale=1, bg=0):
    """
    Draws an elbow (symetric L)
    """
    # create a blank board
    board_np = create_blank_board(scale*2, scale*2, bg)
    # draw the shape on the board
    board_np[0:scale*2, 0:scale] = fg
    board_np[scale:scale*2, scale:scale*2] = fg
    # return the board
    return board_np

def draw_rectangle_shape(fg, scale=1, length=1):
    """
    Draws a rectangle
    """
    # create a blank board
    board_np = create_blank_board(scale, scale*length, 0)
    # draw the shape on the board
    board_np[0:scale, 0:scale*length] = fg
    # return the board
    return board_np

def neighbor_has_color_for_checkerboard(board, x, y, color):
    """
    Checks if a neighbor has a color
    """
    # check if the neighbor has the color
    if x > 0 and board[x-1][y] == color:
        return True
    if x < board.shape[0]-1 and board[x+1][y] == color:
        return True
    if y > 0 and board[x][y-1] == color:
        return True
    if y < board.shape[1]-1 and board[x][y+1] == color:
        return True
    # for i in max(0,x-1), min(board.shape[0]-1,x+2), x+1:
    #     for j in max(0,y-1), min(board.shape[1]-1,y+2), y+1:
    #         if board[i][j] == color:
    #             return True
    # return false if the neighbor does not have the color
    return False

def is_edge_for_checkerboard(board, x, y, fg, bg):
    """
    Checks if a given position is an edge
    If it is fg and x and y are 0 it is an edge
    If it is fg and x and y are at board.shape it is an edge
    If it is fg and has a bg neighbor it is an edge
    If x, y is bg then it is not an edge
    """
    # if the position is bg then it is not an edge
    if board[x, y] == bg:
        return False
    # if the position is fg and x and y are 0 then it is an edge
    if x == 0 or y == 0:
        return True
    # if the position is fg and x and y are at board.shape then it is an edge
    if x == board.shape[0]-1  or y == board.shape[1]-1:
            return True
    # if the position is fg and has a bg neighbor then it is an edge
    if neighbor_has_color_for_checkerboard(board, x, y, bg):
        return True
    # if none of the above conditions are met then it is not an edge
    return False

def in_fill_shape_checkerboard(board, shape_color, in_fill_color):
    """
    Iterate through the board and change shape_color to in_fill_color
    if it is not an edge
    """

    for x in range(board.shape[0]):
        for y in range(board.shape[1]):
            if is_edge_for_checkerboard(board, x, y, shape_color, in_fill_color):
                continue
            if board[x, y] == shape_color:
                board[x, y] = in_fill_color
    return board



def fill_pattern1(board, shape_color, in_fill_color):
    '''
    fill certain parts of the shape with in_fill_color
    '''
    if not isinstance(board, np.ndarray):
        raise TypeError("input must be a numpy board")
    if board.ndim != 2:
        raise ValueError("input must be 2D numpy board")
    if shape_color == in_fill_color:
        raise ValueError("shape color and in fill color cannot be the same")
    shape_color = int(shape_color)
    in_fill_color = int(in_fill_color)
    # find the bounds of the shape
    x_min = np.argmax(board[0, :] == shape_color)
    y_min = np.argmax(board[:, 0] == shape_color)
    x_max = board.shape[1] - np.argmax(board[0, ::-1] == shape_color) - 1
    y_max = board.shape[0] - np.argmax(board[::-1, 0] == shape_color) - 1
    queue = [(x_min, y_min)]

    while len(queue) > 0:
        x, y = queue.pop()

        if board[y, x] == shape_color:
            board[y, x] = in_fill_color
            if x + 1 <= x_max:
                queue.append((x + 1, y))
            if y + 1 <= y_max:
                queue.append((x, y + 1))
            if x - 1 >= x_min:
                queue.append((x - 1, y))
            if y - 1 >= y_min:
                queue.append((x, y - 1))
	
    return board

def shape_color_and_in_fill(board, shape_color, in_fill_color):
    """
    Fills in a shape with a given color and fills in the inside of the shape with a given color
    """
    # select a different color than is in the board
    temp_color = 10
    temp_color2 = 11
    # change all edges to 2
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if is_edge(board, i, j, shape_color, 0):
                board[i, j] = temp_color
    board[board == shape_color] = temp_color2
    board[board == temp_color] = shape_color
    board[board == temp_color2] = in_fill_color
    return board

def neighbor_has_color(board, x, y, color):
    """
    Checks if a neighbor has a color
    """
    # check if the neighbor has the color
    if x > 0 and board[x-1][y] == color:
        return True
    if x < board.shape[0]-1 and board[x+1][y] == color:
        return True
    if y > 0 and board[x][y-1] == color:
        return True
    if y < board.shape[1]-1 and board[x][y+1] == color:
        return True
    # check the diagonals
    if x > 0 and y > 0 and board[x-1][y-1] == color:
        return True
    if x > 0 and y < board.shape[1]-1 and board[x-1][y+1] == color:
        return True
    if x < board.shape[0]-1 and y > 0 and board[x+1][y-1] == color:
        return True
    if x < board.shape[0]-1 and y < board.shape[1]-1 and board[x+1][y+1] == color:
        return True
    # for i in max(0,x-1), min(board.shape[0]-1,x+2), x+1:
    #     for j in max(0,y-1), min(board.shape[1]-1,y+2), y+1:
    #         if board[i][j] == color:
    #             return True
    # return false if the neighbor does not have the color
    return False

def is_edge(board, x, y, fg, bg):
    """
    Checks if a given position is an edge
    If it is fg and x and y are 0 it is an edge
    If it is fg and x and y are at board.shape it is an edge
    If it is fg and has a bg neighbor it is an edge
    If x, y is bg then it is not an edge
    """
    # if the position is bg then it is not an edge
    if board[x, y] != fg:
        return False
    # if the position is fg and x and y are 0 then it is an edge
    if x == 0 or y == 0:
        return True
    # if the position is fg and x and y are at board.shape then it is an edge
    if x == board.shape[0]-1  or y == board.shape[1]-1:
        return True
    # if the position is fg and has a bg neighbor then it is an edge
    if neighbor_has_color(board, x, y, bg):
        return True
    # if none of the above conditions are met then it is not an edge
    return False

def fill_pattern2(board, shape_color, in_fill_color):
    board = fill_pattern1(board, shape_color, in_fill_color)
    board = shape_color_and_in_fill(board, shape_color, in_fill_color)
    return board

def draw_shape(board, shape_fn, bring_to_front: bool, fg: int, startx: int, starty: int, scale=1, cardinality=0, bg=-1, in_fill=-1, in_fill_function=None):
    height=board.shape[0]
    width=board.shape[1]
    board_np = np.copy(board)
    # run the shape function to draw the shape on the board overlay
    # print the function name and scale
    #print(f"shape_fn.__name__ {shape_fn.__name__}, scale={scale}, cardinality={cardinality}, startx={startx}, starty={starty}, in_fill={in_fill}, in_fill_function.__name__={in_fill_function}.__name__")
    shape_board_np = np.rot90(remove_padding(shape_fn(fg, scale, bg)), k=cardinality)
    if in_fill != -1:
        shape_board_np=in_fill_function(shape_board_np, fg, in_fill)
    #plot_board(shape_board_np)
    # check to see if startx or y are negative and add rows or columns to the board if needed
    if startx < 0:
        board_np = insert_column(board_np, abs(startx), 0)
        startx = 0
    if starty < 0:
        board_np = insert_row(board_np, abs(starty), 0)
        starty = 0
    overlay = np.zeros((max(starty+shape_board_np.shape[0], height), max(startx+shape_board_np.shape[1], width)), dtype=np.int8)
    
    overlay[starty:starty+shape_board_np.shape[0], startx:startx+shape_board_np.shape[1]] = shape_board_np
    if bg==-1:
        bg_color = get_board_bg_color(board)
    else:
        bg_color = bg
    #print(bg_color)
    mask = np.ones((overlay.shape[0], overlay.shape[1]), dtype=np.int8)

    # make overlay the same dimensions as the board (expand the overlay if needed and expand the board if needed)
    if height < overlay.shape[0] or width < overlay.shape[1]:
        # losing the pixel on the next line
        board_np = resize_array_padding_right_and_bottom(board_np, overlay, bg_color)
        height=board_np.shape[0]
        width=board_np.shape[1]
    if height > overlay.shape[0] or width > overlay.shape[1]:
        overlay = resize_array_padding_right_and_bottom(overlay, board_np, bg_color)
        mask = resize_array_padding_right_and_bottom(mask, board_np, 1)
    # if bring to front is true then change the mask values to 0 where the board_np values are not 0
    if bring_to_front:
        mask[overlay!=bg_color] = 0
    else:
        overlay[board_np!=bg_color] = 0
    res = board_np * mask + overlay
    return res

def insert_row(board, row_count=1, row_index=0, bg_color=0):
    """
    Inserts a row at the specified index
    """
    # create a blank board
    board_np = np.copy(board)
    # insert the row(s)
    for i in range(row_count):
        board_np = np.insert(board_np, row_index, bg_color, axis=0)
    # return the board
    return board_np

def insert_column(board, column_count=1, column_index=0, bg_color=0):
    """
    Inserts a column at the specified index
    """
    # create a blank board
    board_np = np.copy(board)
    # insert the column(s)
    for i in range(column_count):
        board_np = np.insert(board_np, column_index, bg_color, axis=1)
    # return the board
    return board_np

def remove_row(board, row_count=1, row_index=0):
    """
    Removes a row at the specified index
    """
    # create a blank board
    board_np = np.copy(board)
    # remove the row(s)
    for i in range(row_count):
        board_np = np.delete(board_np, row_index, axis=0)
    # return the board
    return board_np

def remove_column(board, column_count=1, column_index=0):
    """
    Removes a column at the specified index
    """
    # create a blank board
    board_np = np.copy(board)
    # remove the column(s)
    for i in range(column_count):
        board_np = np.delete(board_np, column_index, axis=1)
    # return the board
    return board_np

def add_row(board, row_count=1, bg_color=0):
    """
    Adds a row to the end of the board
    """
    # create a blank board
    board_np = np.copy(board)
    print(board_np.shape[1])
    # add the row(s)
    for i in range(row_count):
        board_np = np.append(board_np, [[bg_color]*board_np.shape[1]], axis=0)
    # return the board
    return board_np

def add_column(board, column_count=1, bg_color=0):
    """
    Adds a column to the end of the board
    """
    # create a blank board
    board_np = np.copy(board)
    # add the column(s)
    new_col = np.array([[bg_color]*board_np.shape[0]]).T
    print(f"adding column with shape {new_col.shape} with {board_np.shape[0]} items ")
    print([[bg_color]*board_np.shape[0]])

    for i in range(column_count):
        board_np = np.append(board_np, new_col, axis=1)
    # return the board
    return board_np

def scale_board(board, scale_factor, center_content=False, bg_color=0):
    """
    Scales the board by the specified scale factor (manually, do not use cv2.resize)
    """
    # create a blank board
    board_np = np.copy(board)
    # scale the board
    scaled_board = np.array([[bg_color]*int(board_np.shape[1]*scale_factor)]*int(board_np.shape[0]*scale_factor))
    if center_content:
        # center the content
        scaled_board[int((scaled_board.shape[0]-board_np.shape[0])/2):int((scaled_board.shape[0]-board_np.shape[0])/2)+board_np.shape[0], int((scaled_board.shape[1]-board_np.shape[1])/2):int((scaled_board.shape[1]-board_np.shape[1])/2)+board_np.shape[1]] = board_np
    else:
        scaled_board[:board_np.shape[0], :board_np.shape[1]] = board_np
    # return the board
    return scaled_board

def resize_board(board, new_width, new_height, crop=True, center_content=False, bg_color=0):
    """
    Resizes the board to the specified width and height (manually, do not use cv2.resize)
    If not cropping, the board will not scale down to fit the new size, but it will scale up to fit the new size
    """
    return_board = board
    if new_width > board.shape[1] and new_height > board.shape[0]:
        # create a blank board
        board_np = np.copy(board)
        # resize the board
        resized_board = np.array([[bg_color]*new_width]*new_height)
        if center_content:
            # center the content
            resized_board[int((resized_board.shape[0]-board_np.shape[0])/2):int((resized_board.shape[0]-board_np.shape[0])/2)+board_np.shape[0], int((resized_board.shape[1]-board_np.shape[1])/2):int((resized_board.shape[1]-board_np.shape[1])/2)+board_np.shape[1]] = board_np
        else:
            resized_board[:board_np.shape[0], :board_np.shape[1]] = board_np
        # return the board
        return_board = resized_board
    elif crop and (new_width < board.shape[1] or new_height < board.shape[0]):
        # create a blank board
        board_np = np.copy(board)
        # downsize the board
        if new_width < board.shape[1]:
            # crop the width
            if center_content:
                # center the content
                board_np = board_np[:, int((board_np.shape[1]-new_width)/2):int((board_np.shape[1]-new_width)/2)+new_width]
            else:
                board_np = board_np[:, :new_width]
        if new_height < board.shape[0]:
            # crop the height
            if center_content:
                # center the content
                board_np = board_np[int((board_np.shape[0]-new_height)/2):int((board_np.shape[0]-new_height)/2)+new_height, :]
            else:
                board_np = board_np[:new_height, :]
        resized_board = np.array([[bg_color]*new_width]*new_height)
        if center_content:
            # center the content
            resized_board[int((resized_board.shape[0]-board_np.shape[0])/2):int((resized_board.shape[0]-board_np.shape[0])/2)+board_np.shape[0], int((resized_board.shape[1]-board_np.shape[1])/2):int((resized_board.shape[1]-board_np.shape[1])/2)+board_np.shape[1]] = board_np
        else:
            resized_board[:board_np.shape[0], :board_np.shape[1]] = board_np
        # return the board
        return_board = resized_board
    else:
        new_height = max(new_height, board.shape[0])
        new_width = max(new_width, board.shape[1])
        # create a blank board
        board_np = np.copy(board)
        # resize the board
        resized_board = np.array([[bg_color]*new_width]*new_height)
        if center_content:
            # center the content
            resized_board[int((resized_board.shape[0]-board_np.shape[0])/2):int((resized_board.shape[0]-board_np.shape[0])/2)+board_np.shape[0], int((resized_board.shape[1]-board_np.shape[1])/2):int((resized_board.shape[1]-board_np.shape[1])/2)+board_np.shape[1]] = board_np
        # return the board
        return_board = resized_board
    return return_board

def remove_padding(board, bg_color=0):
    """
    Removes the padding from the board
    """
    # create a blank board
    board_np = np.copy(board)
    # remove the padding
    board_np = board_np[~np.all(board_np == bg_color, axis=1)]
    board_np = board_np[:, ~np.all(board_np == bg_color, axis=0)]
    # return the board
    return board_np

def add_padding(board, padding_size, bg_color=0):
    """
    Adds padding to the board
    """
    # create a blank board
    board_np = np.copy(board)
    # add the padding
    board_np = np.pad(board_np, padding_size, constant_values=bg_color)
    # return the board
    return board_np

def get_lowest_content_index(board, bg_color=0):
    """
    Returns the index of the lowest content
    """
    lowest_index = 0
    # get the lowest content index
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i][j] != bg_color:
                if lowest_index < i:
                    lowest_index = i
    return lowest_index

def get_rightmost_content_index(board, bg_color=0):
    """
    Returns the index of the rightmost content
    """
    rightmost_index = 0
    # get the rightmost content index
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i][j] != bg_color:
                if j > rightmost_index:
                    rightmost_index = j
    return rightmost_index

def resize_array_padding_right_and_bottom(array_to_reshape, array_to_copy_size_from, pad_with=0):
    """
    This only works if the array_to_reshape is smaller than array_to_copy_size_from.
    Resize numpy ndarray array_to_reshape to be the same size as array_to_copy_size_from.
    The new cells should be pad_with and added to the right and bottom of ndarray array_to_reshape.
    """
    # get the dimensions of array_to_reshape
    dimensions = array_to_reshape.shape

    # get the dimensions of array_to_copy_size_from
    compare_dimensions = array_to_copy_size_from.shape

    # array to hold the new larger array
    new_array = np.zeros((max(compare_dimensions[0], dimensions[0]), max(compare_dimensions[1], dimensions[1])), dtype=array_to_reshape.dtype)
    new_array.fill(pad_with)

    # copy the smaller array into the larger array starting at x,y coordinates
    # only resize if the array_to_reshape is smaller than array_to_copy_size_from
    if dimensions[0] > compare_dimensions[0] or dimensions[1] > compare_dimensions[1]:
        if dimensions[0] > compare_dimensions[0] and dimensions[1] > compare_dimensions[1]:
            new_array[0:compare_dimensions[0], 0:compare_dimensions[1]] = array_to_reshape[0:compare_dimensions[0], 0:compare_dimensions[1]]
        else:
            new_array[0:dimensions[0],0:dimensions[1]] = array_to_reshape


        # elif dimensions[0] > compare_dimensions[0]:
        #     new_array[0:compare_dimensions[0], 0:dimensions[1]] = array_to_reshape[0:compare_dimensions[0], 0:dimensions[1]]
        # elif dimensions[1] > compare_dimensions[1]:
        #     new_array[0:dimensions[0],0:dimensions[1]] = array_to_reshape
    else: 
        new_array[0:dimensions[0],0:dimensions[1]] = array_to_reshape

    # return the new larger array
    return new_array

def draw_line_xy_to_xy(board, bring_to_front: bool, fg: int, startx: int, starty: int, endx: int, endy: int, bg=-1):
    #print(f"draw_line_xy_to_xy {startx} {starty} {endx} {endy}")
    line_colors = []
    if bg==-1:
        bg_color = get_board_bg_color(board)
    else:
        bg_color = bg
    height=board.shape[0]
    width=board.shape[1]
    board_np = board.copy()
    if startx < endx:
        startx = startx-1
        starty = starty-1
        endx = endx-1
        endy = endy-1

    #create a blank board
    overlay = create_blank_board(max(max(starty, endy)+1,1), max(max(startx, endx)+1,1), bg)
    newfg=fg if fg>-1 else np.random.randint(1, 10)
    # draw the line on the board overlay by developing the line equation
    if startx == endx:
        #print("vertical line")
        # vertical line
        if starty < endy:
            for y in range(starty, endy):
                overlay[y][max(startx,0)] = newfg
                line_colors.append(newfg)
                newfg=fg if fg>-1 else np.random.randint(1, 10)
        else:
            for y in range(endy+1, starty+1):
                overlay[y][max(startx,0)] = newfg
                #line_colors.append(newfg)
                line_colors.insert(0, newfg)
                newfg=fg if fg>-1 else np.random.randint(1, 10)
    elif starty == endy:
        # horizontal line
        if startx < endx:
            for x in range(startx+1, endx+1):
                overlay[starty][x] = newfg
                line_colors.append(newfg)
                newfg=fg if fg>-1 else np.random.randint(1, 10)
        else: 
            for x in range(endx+1, startx+1):
                overlay[starty][x] = newfg
                # insert at the front of the list
                line_colors.insert(0, newfg)
                newfg=fg if fg>-1 else np.random.randint(1, 10)
    else:
        slope = (endy-starty)/(endx-startx)
        #print(f"slope: {slope}")
        startx = max(startx, 0)
        starty = max(starty, 0)
        endx = max(endx, 0)
        endy = max(endy, 0)
        
        x = startx
        y = starty
        step = 1 if startx < endx else -1
        while (step==-1 and ((slope>0 and x!=endx-1) or (slope<0 and x!=endx))) or (step==1 and x!=endx+1):
            y = round(slope*(x-startx)+starty)
            #print(f"x: {x}, y: {y}")
            #print(f"overlay.shape: {overlay.shape}")
            overlay[y][x] = fg if fg>-1 else np.random.randint(1, 10)
            line_colors.append(overlay[y][x])
            x += step
    # make overlay the same dimensions as the board (expand the overlay if needed and expand the board if needed)
    if height < overlay.shape[0] or width < overlay.shape[1]:
        board_np = resize_array_padding_right_and_bottom(board_np, overlay, bg_color)
        height=board_np.shape[0]
        width=board_np.shape[1]
    if height > overlay.shape[0] or width > overlay.shape[1]:
        overlay = resize_array_padding_right_and_bottom(overlay, board_np, bg_color)
    # if bring to front is true then change the overlay values to 0 where the board_np values are not 0
    if bring_to_front:
        board_np[overlay!=bg_color] = 0
    else:
        overlay[board_np!=bg_color] = bg_color
    res = board_np + overlay

    return line_colors, res

def draw_line_from_point(board, bring_to_front: bool, fg: int, startx: int, starty: int, cardinality8: int, length: int, bg=0):
    '''
    cardinality8 is 0-7 where 0 is up, 1 is up right, 2 is right, 3 is down right, 4 is down, 5 is down left, 6 is left, 7 is up left
    use the function draw_line_xy_to_xy to draw a line from a point to another point
    '''
    line_colors = []
    #print(f"draw_line_from_point {startx} {starty} {cardinality8} {length}")
    if bg==-1:
        bg_color = get_board_bg_color(board)
    else:
        bg_color = bg
    if cardinality8 == 0:
        # up
        board, line_colors = draw_line_xy_to_xy(board, bring_to_front, fg, startx, starty, startx, starty-length, bg_color)
    elif cardinality8 == 1:
        # up right
        board, line_colors = draw_line_xy_to_xy(board, bring_to_front, fg, startx, starty+1, startx+length, starty-length+1, bg_color)
    elif cardinality8 == 2:
        # right
        board, line_colors = draw_line_xy_to_xy(board, bring_to_front, fg, startx, starty+1, startx+length, starty+1, bg_color)
    elif cardinality8 == 3:
        # down right
        board, line_colors = draw_line_xy_to_xy(board, bring_to_front, fg, startx, starty, startx+length, starty+length, bg_color)
    elif cardinality8 == 4:
        # down
        board, line_colors = draw_line_xy_to_xy(board, bring_to_front, fg, startx, starty, startx, starty+length, bg_color)
    elif cardinality8 == 5:
        # down left
        board, line_colors = draw_line_xy_to_xy(board, bring_to_front, fg, startx, starty, startx-length, starty+length, bg_color)
    elif cardinality8 == 6:
        # left
        board, line_colors = draw_line_xy_to_xy(board, bring_to_front, fg, startx, starty, startx-length, starty, bg_color)
    elif cardinality8 == 7:
        # up left
        xend = startx-length+1 if length > 1 else startx-length
        yend = starty-length+1 if length > 1 else starty-length
        board, line_colors = draw_line_xy_to_xy(board, bring_to_front, fg, startx, starty, xend, yend, bg_color)
    return line_colors, board

def get_shape_dims_by_cardinality(shape_fn, cardinality):
    '''
    create a blank board
    draw the shape specified by shape_fn
    return the width and height of the shape
    '''
    board = create_blank_board(1, 1, 0)
    board = draw_shape(board, shape_fn, True, 1, 0, 0, 1, cardinality, 0)
    return board.shape


def draw_repeating_shapes_in_line_f_point(board, bring_to_front: bool, colors: list, startx: int, starty: int, line_cardinality8: int, length: int, shape: str, shape_cardinality_start=-1, bg=0, inter_shape_spacing=1, scale=1, in_fill=-1, in_fill_function=None):
    '''
    line_cardinality8 is 0-7 where 0 is up, 1 is up right, 2 is right, 3 is down right, 4 is down, 5 is down left, 6 is left, 7 is up left
    use the function draw_line_xy_to_xy to draw a line from a point to another point
    '''
    shape_cardinalities = []
    current_cardinality = shape_cardinality_start
    if current_cardinality > -1:
        for i in range(length):
            shape_cardinalities.append(shape_cardinality_start+i)
            current_cardinality += 1
            if current_cardinality > 3:
                current_cardinality = 0
    else:
        for i in range(length):
            shape_cardinalities.append(0)
    #print(f"draw_line_from_point {startx} {starty} {line_cardinality8} {length}")
    if bg==-1:
        bg_color = get_board_bg_color(board)
    else:
        bg_color = bg
        #(board, shape_fn, bring_to_front: bool, fg: int, startx: int, starty: int, scale=1, cardinality=0, bg=-1)

    if line_cardinality8 == 7 or line_cardinality8 == 5 or line_cardinality8 == 6 or line_cardinality8 == 4:
        colors=np.flip(colors)
    
    board = draw_shape(board, shape, bring_to_front, colors[0], 0, 0, scale, shape_cardinalities[0], 0, in_fill=in_fill, in_fill_function=in_fill_function)
    
    for i in range(1, length):
        # 3 and 7 are the same - just reverse the colors
        # 5 and 1 are the same - just reverse the colors
        if line_cardinality8 == 1 or line_cardinality8 == 5:
            #print(f"shape height {get_shape_dims_by_cardinality(shape, shape_cardinalities[i])[0]*scale}, shape width {get_shape_dims_by_cardinality(shape, shape_cardinalities[i])[1]*scale}")
            #print(f"should draw at y {0-get_shape_dims_by_cardinality(shape, shape_cardinalities[i])[0]-inter_shape_spacing}")
    #def draw_shape(board, shape_fn, bring_to_front: bool, fg: int, startx: int, starty: int, scale=1, cardinality=0, bg=-1):
            board = draw_shape(board, shape, bring_to_front, colors[i], get_rightmost_content_index(board, 0)+1+inter_shape_spacing, 0-get_shape_dims_by_cardinality(shape, shape_cardinalities[i])[0]*scale-inter_shape_spacing, scale, shape_cardinalities[i], 0, in_fill=in_fill, in_fill_function=in_fill_function)
        elif line_cardinality8 == 3 or line_cardinality8 == 7:
            #print(f"colors {colors}, line_cardinality8 {line_cardinality8}, shape_cardinalities {shape_cardinalities[i]}")
            board = draw_shape(board, shape, bring_to_front, colors[i], get_rightmost_content_index(board, 0)+1+inter_shape_spacing, get_lowest_content_index(board, 0)+1+inter_shape_spacing, scale, shape_cardinalities[i], 0, in_fill=in_fill, in_fill_function=in_fill_function)
        elif line_cardinality8 == 2 or line_cardinality8 == 6: # horizontal
            board = draw_shape(board, shape, bring_to_front, colors[i], get_rightmost_content_index(board, 0)+1+inter_shape_spacing, 0, scale, shape_cardinalities[i], 0, in_fill=in_fill, in_fill_function=in_fill_function)
        elif line_cardinality8 == 4 or line_cardinality8 == 0: # vertical
            board = draw_shape(board, shape, bring_to_front, colors[i], 0, get_lowest_content_index(board, 0)+1+inter_shape_spacing, scale, shape_cardinalities[i], 0, in_fill=in_fill, in_fill_function=in_fill_function)
    return board
    
def get_start_xy_for_line_f_corner(board_dims, corner):
    if corner == 0:
        return 0, 0
    elif corner == 1:
        return board_dims[1]-1, 0
    elif corner == 2:
        return board_dims[1]-1, board_dims[0]-1
    elif corner == 3:
        return 0, board_dims[0]-1
    else:
        raise Exception(f"Invalid corner: {corner}")

def get_cardinality8(riddle_type, corner, line_length, scale):
    if riddle_type == "color_rotate":
        line_length = corner+1 # use the line length to store the cardinality
        corner = np.random.randint(0,4)
    if corner == 0:
        if line_length > scale or line_length == 1:
            cardinality8 = np.random.choice([2, 4])
        else:
            cardinality8 = np.random.choice([2, 3, 4])
    elif corner == 1:
        if line_length > scale or line_length == 1:
            cardinality8 = np.random.choice([4, 6])
        else:
            cardinality8 = np.random.choice([4, 5, 6])
    elif corner == 2:
        if line_length > scale or line_length == 1:
            cardinality8 = np.random.choice([6, 0])
        else:
            cardinality8 = np.random.choice([0, 6, 7])
    elif corner == 3:
        if line_length > scale or line_length == 1:
            cardinality8 = np.random.choice([0, 2])
        else:
            cardinality8 = np.random.choice([0, 1, 2])
    return cardinality8, line_length

def create_shape_scaling_rid(riddle_type, num_test_items, shape_functions, train_corners, test_corner, cardinality4_mapping, reverse_input_output=False, rows=0, cols=0, indicator_count=1, indicator_color_mode="color", color_shape1=0, shape_scale1=0, debug_mode=False):
    """
    Creates a shape, color, rotation, object, scaling riddle in one of many riddle_types
    """

    input_grids = []
    output_grids = []
    params = {}
    params['input'] = []
    params['output'] = []
    color_bg = 0

    x = 1 if cols == 0 else cols
    y = 1 if rows == 0 else rows
    num_test_items = np.random.randint(1, 7, dtype=int) if num_test_items == 0 else num_test_items
    for i in range(num_test_items):
        color_shape = np.random.randint(1,10)
        color_line = np.random.randint(1,10)
        while color_line == color_shape or color_line == color_shape1:
            color_line = np.random.randint(1,10)

        # create the input grid
        # set the parameters for the input and output grids
        working_board = create_blank_board(x, y, color_bg)
        # pick a random shape
        shape = random.choice(shape_functions)
        scale = np.random.randint(1, 6) if shape_scale1 == 0 else shape_scale1

        if "random_indicator_color" in riddle_type:
            color_line = -1
        working_board = draw_shape(working_board, shape, True, color_shape, 0, 0, scale, 0, 0)
        working_board = remove_padding(working_board, 0)
        # add padding
        working_board = add_padding(working_board, scale, color_bg)

        # choose length of line between 1 and 5
        line_length = np.random.randint(1, 6)
        while line_length == shape_scale1 or line_length >= working_board.shape[0] or line_length >= working_board.shape[1]:
            line_length = np.random.randint(1, 6)

        #if (riddle_type=="reverse_count_color_and_repeat_shape_bg_input_shape_color" and i==0) or riddle_type!="reverse_count_color_and_repeat_shape_bg_input_shape_color":
        # choose a corner to add a line
        corner = train_corners[i] if i < num_test_items-1 else test_corner

        
        # set the line length to cardinality for color_rotation
        if riddle_type == "color_rotate":
            line_length = corner+1 # use the line length to store the cardinality
            corner = np.random.randint(0,4)

        x = working_board.shape[1]-1
        y = working_board.shape[0]-1
        # set cardinality8 and the x, y coordinates for the line
        cardinality8, line_length = get_cardinality8(riddle_type, corner, line_length, scale)

        x, y = get_start_xy_for_line_f_corner(working_board.shape, corner)
        if debug_mode:
            print(f"corner: {corner}, x: {x}, y: {y}, line_length: {line_length}, cardinality8: {cardinality8}")
        working_board, line_colors = draw_line_from_point(working_board, True, color_line, x, y, cardinality8, line_length, 0)
 
        if debug_mode:           
            print(f"corner: {corner}, cardinality8: {cardinality8}, line_length: {line_length}, cardinality4: {cardinality4_mapping}, corner: {corner}")

        working_board2 = create_blank_board(1, 1, color_bg)
        # draw a left L scale 1
        if riddle_type == "color_scale_rotate_remove_padding" or riddle_type == "color_scale_rotate":
            working_board2 = draw_shape(working_board2, shape, True, color_line, 0, 0, line_length, cardinality4_mapping[corner], color_bg)
            working_board2 = remove_padding(working_board2, 0)
            if riddle_type == "color_scale_rotate":
                working_board2=add_padding(working_board2, line_length, color_bg)
        elif riddle_type == "color_scale":
            working_board2 = draw_shape(working_board2, shape, True, color_line, 0, 0, line_length, 0, color_bg)
            working_board2 = remove_padding(working_board2, 0)
            working_board2=add_padding(working_board2, line_length, color_bg)
        elif riddle_type == "color_scale_remove_padding":
            working_board2 = draw_shape(working_board2, shape, True, color_line, 0, 0, line_length, 0, color_bg)
            working_board2 = remove_padding(working_board2, 0)
        elif riddle_type == "padding_f_shape_scale_shape_scale_f_indicator_rotate":
            working_board2 = draw_shape(working_board2, shape, True, color_line, 0, 0, line_length, cardinality4_mapping[corner], color_bg)
            working_board2 = remove_padding(working_board2, 0)
            working_board2=add_padding(working_board2, scale, color_shape)
        elif riddle_type == "color_rotate":
            working_board2 = draw_shape(working_board2, shape, True, color_line, 0, 0, scale,  cardinality4_mapping[line_length-1], color_bg)
            working_board2 = remove_padding(working_board2, 0)
            working_board2=add_padding(working_board2, scale, color_bg)
        elif riddle_type == "scale_rotate":
            working_board2 = draw_shape(working_board2, shape, True, color_shape, 0, 0, line_length, cardinality4_mapping[corner], color_bg)
            working_board2 = remove_padding(working_board2, 0)
            working_board2=add_padding(working_board2, scale, color_bg)
        elif riddle_type == "color":
            working_board2 = draw_shape(working_board2, shape, True, color_line, 0, 0, scale, 0, color_bg)
            working_board2 = remove_padding(working_board2, 0)
            working_board2=add_padding(working_board2, scale, color_bg)
        elif riddle_type == "rotate":
            working_board2 = draw_shape(working_board2, shape, True, color_shape, 0, 0, scale, cardinality4_mapping[corner], color_bg)
            working_board2 = remove_padding(working_board2, 0)
            working_board2=add_padding(working_board2, scale, color_bg)
        elif riddle_type == "scale_to_padding_line_length_to_scale_random_indicator_color":
            working_board2 = draw_shape(working_board2, shape, True, color_shape, 0, 0, line_length, 0, color_bg)
            working_board2 = remove_padding(working_board2, 0)
            working_board2=add_padding(working_board2, scale, color_bg)
        elif riddle_type == "scale_to_padding_line_length_to_scale_random_indicator_color_corner_to_rotate":
            working_board2 = draw_shape(working_board2, shape, True, color_shape, 0, 0, line_length, cardinality4_mapping[corner], color_bg)
            working_board2 = remove_padding(working_board2, 0)
            working_board2=add_padding(working_board2, scale, color_bg)
        elif riddle_type == "scale_to_padding_line_length_to_scale_random_indicator_color_corner_to_rotate_padding_color_f_shape_shape_color_f_first_line_pixel":
            working_board2 = draw_shape(working_board2, shape, True, line_colors[0], 0, 0, line_length, cardinality4_mapping[corner], color_bg)
            working_board2 = remove_padding(working_board2, 0)
            working_board2=add_padding(working_board2, scale, color_shape)
        elif riddle_type == "color_scale_rotate_in_fill_pattern1":
            working_board2 = draw_shape(working_board2, shape, True, color_line, 0, 0, line_length, cardinality4_mapping[corner], color_bg, color_shape, fill_pattern1) # shape_color_and_in_fill
            working_board2 = remove_padding(working_board2, 0)
            working_board2=add_padding(working_board2, scale, 0)
        elif riddle_type == "color_scale_rotate_in_fill_pattern2":
            working_board2 = draw_shape(working_board2, shape, True, color_line, 0, 0, line_length, cardinality4_mapping[corner], color_bg, color_shape, fill_pattern2)
            working_board2 = remove_padding(working_board2, 0)
            working_board2=add_padding(working_board2, scale, 0)
        elif riddle_type == "color_scale_rotate_in_fill":
            working_board2 = draw_shape(working_board2, shape, True, color_line, 0, 0, line_length, cardinality4_mapping[corner], color_bg, color_shape, shape_color_and_in_fill)
            #working_board2 = draw_shape(working_board2, shape, True, 1, 0, 0, scale, 0, 0, 2, shape_color_and_in_fill)
            working_board2 = remove_padding(working_board2, 0)
            working_board2=add_padding(working_board2, scale, 0)
        elif riddle_type == "color_scale_rotate_in_fill_checkerboard":
            working_board2 = draw_shape(working_board2, shape, True, color_line, 0, 0, line_length, cardinality4_mapping[corner], color_bg, color_shape, in_fill_shape_checkerboard)
            working_board2 = remove_padding(working_board2, 0)
            working_board2=add_padding(working_board2, scale, 0)
        if reverse_input_output:
            input_grids.append(working_board2)
            output_grids.append(working_board)
        else:
            input_grids.append(working_board)
            output_grids.append(working_board2)
        params['input'].append({'shape': shape.__name__, 'scale': scale, 'color': color_shape, 'padding': scale, 'corner': corner, 'line_length': line_length, 'line_color': color_line, "line_colors": line_colors, "cardinality8": cardinality8})
        params['output'].append({'shape': shape.__name__, 'scale': line_length, 'color': color_line, 'padding': 0, 'corner': corner, 'line_length': line_length, 'line_color': color_line})
    return input_grids, output_grids, params  


def create_shape_repetition_riddle(riddle_type, num_test_items, shape_functions, train_corners, test_corner, cardinality4_mapping, reverse_input_output=False, rows=0, cols=0, indicator_count=1, indicator_color_mode="color", color_shape1=0, shape_scale1=0, debug_mode=False):
    """
    Creates a shape, color, rotation, object, scaling riddle in one of many riddle_types
    """

    input_grids = []
    output_grids = []
    params = {}
    params['input'] = []
    params['output'] = []
    color_bg = 0
    # set the parameters for the input and output grids
    x = 1 if cols == 0 else cols
    y = 1 if rows == 0 else rows
    num_test_items = np.random.randint(1, 7, dtype=int) if num_test_items == 0 else num_test_items
    for i in range(num_test_items):
        color_shape = np.random.randint(1,10)
        color_line = np.random.randint(1,10)
        while color_line == color_shape or color_line == color_shape1:
            color_line = np.random.randint(1,10)

        # create the input grid
        working_board = create_blank_board(1, 1, color_bg)
        # pick a random shape
        shape = random.choice(shape_functions)
        if riddle_type=="reverse_count_color_and_repeat_shape_bg_input_shape_color":
            scale = 1
        else:
            scale = np.random.randint(1, 6) if shape_scale1 == 0 else shape_scale1

        working_board = draw_shape(working_board, shape, True, color_shape, 0, 0, scale, 0, 0)
        #working_board = remove_padding(working_board, 0)
        # add padding
        working_board = add_padding(working_board, scale, color_bg)

        # choose length of line between 1 and 5
        line_length = np.random.randint(1, 6)
        while line_length == shape_scale1 or line_length >= working_board.shape[0] or line_length >= working_board.shape[1]:
            line_length = np.random.randint(1, 6)

        if (riddle_type=="reverse_count_color_and_repeat_shape_bg_input_shape_color" and i==0) or riddle_type!="reverse_count_color_and_repeat_shape_bg_input_shape_color":
            # choose a corner to add a line
            corner = train_corners[i] if i < num_test_items-1 else test_corner
            x = working_board.shape[1]-1
            y = working_board.shape[0]-1
            # set cardinality8 and the x, y coordinates for the line
            cardinality8, line_length = get_cardinality8(riddle_type, corner, line_length, scale)

        x, y = get_start_xy_for_line_f_corner(working_board.shape, corner)
        if debug_mode:
            print(f"corner: {corner}, x: {x}, y: {y}, line_length: {line_length}, cardinality8: {cardinality8}")
        
        if "random_line_colors" in riddle_type:
            color_line = -1

        working_board, line_colors = draw_line_from_point(working_board, True, color_line, x, y, cardinality8, line_length, 0)
 
        if debug_mode:           
            print(f"corner: {corner}, cardinality8: {cardinality8}, line_length: {line_length}, cardinality4: {cardinality4_mapping}, corner: {corner}")
        working_board2 = create_blank_board(1, 1, 0)
        # draw a left L scale 1
        if riddle_type == "count_color_and_repeat_shape":
            for i in range(line_length):
                working_board2 = draw_shape(working_board2, shape, True, color_line, get_rightmost_content_index(working_board2)+2, 0, 1, 0, 0)
            working_board2 = remove_padding(working_board2, 0)
        elif riddle_type == "count_color_and_repeat_shape_bg_input_shape_color" or riddle_type == "reverse_count_color_and_repeat_shape_bg_input_shape_color":
            for i in range(line_length):
                working_board2 = draw_shape(working_board2, shape, True, color_line, get_rightmost_content_index(working_board2, 0)+2, 0, 1, 0, 0)
            working_board2 = remove_padding(working_board2, 0)
            # replace all zeros in working_board2 with color_shape to set the background color
            working_board2[working_board2 == 0] = color_shape
        elif riddle_type == "padding_f_scale_and_shape_repetition_f_line_length_and_direction_f_line_direction_shape_scale_1_shape_color_f_line":
            working_board2=draw_repeating_shapes_in_line_f_point(working_board2, True, line_colors, 0, 0, cardinality8, line_length, shape, -1, bg=0, inter_shape_spacing=1)
            working_board2[working_board2 == 0] = color_shape
            working_board2=add_padding(working_board2, scale, color_shape)
        elif riddle_type == "padding_f_scale_and_shape_repetition_f_line_length_and_direction_f_line_direction_and_rotate_starting_with_cardinality_f_corner_adding_90_degrees_per_shape_repetition_shape_scale_1_shape_color_f_line":
            working_board2=draw_repeating_shapes_in_line_f_point(working_board2, True, line_colors, 0, 0, cardinality8, line_length, shape, cardinality4_mapping[corner], bg=0, inter_shape_spacing=scale)

            # working_board2 = draw_shape(working_board2, shape, True, color_line, 0, 0, 1, 0, 0)
            # right_start = get_rightmost_content_index(working_board2, 0)+2
            # bottom_start = get_lowest_content_index(working_board2, 0)+2
            # mid_start = int(bottom_start / 2)
            # for i in range(line_length-1):
            #     if cardinality8 == 1 or cardinality8 == 3 or cardinality8 == 5 or cardinality8 == 7:
            #         working_board2 = draw_shape(working_board2, shape, True, color_line, get_rightmost_content_index(working_board2, 0)+2, get_lowest_content_index(working_board2, 0)+2, 1, 0, 0)
            #     elif cardinality8 == 2 or cardinality8 == 6: # horizontal
            #         working_board2 = draw_shape(working_board2, shape, True, color_line, get_rightmost_content_index(working_board2, 0)+2, 0, 1, 0, 0)
            #     elif cardinality8 == 4 or cardinality8 == 0: # vertical
            #         working_board2 = draw_shape(working_board2, shape, True, color_line, 0, get_lowest_content_index(working_board2, 0)+2, 1, 0, 0)



                #working_board2 = remove_padding(working_board2, 0)
                #working_board2 = draw_shape(working_board2, shape, True, color_line, get_rightmost_content_index(working_board2, 0)+2, 0, 1, 0, 0)
            #working_board2 = remove_padding(working_board2, 0)
            # replace all zeros in working_board2 with color_shape to set the background color
            working_board2[working_board2 == 0] = color_shape
            # add padding
            working_board2=add_padding(working_board2, scale, color_shape)
        elif riddle_type == "padding_f_scale_shape_repeat_f_line_length_direction_f_line_direction_rotate_starting_with_cardinality_f_corner_adding_90_degrees_per_shape_repeat_shape_color_f_line_shape_scale_1_inter_shape_distance_f_scale_random_line_colors":
            working_board2=draw_repeating_shapes_in_line_f_point(working_board2, True, line_colors, 0, 0, cardinality8, line_length, shape, cardinality4_mapping[corner], bg=0, inter_shape_spacing=scale)
            # replace all zeros in working_board2 with color_shape to set the background color
            working_board2[working_board2 == 0] = color_shape
            # add padding
            working_board2=add_padding(working_board2, scale, color_shape)
        elif riddle_type == "no_padding_scale_f_line_length_rotate_starting_with_cardinality_f_corner_adding_90_degrees_per_shape_repetition_color_f_last_color_in_line_direction_f_line_direction_inter_item_distance_of_0":
            # set all elements in line_colors to the value in the last element
            shape_colors = np.full(scale, line_colors[-1])
            #draw_repeating_shapes_in_line_f_point(board, bring_to_front: bool, colors: list, startx: int, starty: int, line_cardinality8: int, length: int, shape: str, shape_cardinality_start=-1, bg=0, inter_shape_spacing=1, scale=1):
            working_board2=draw_repeating_shapes_in_line_f_point(working_board2, True, shape_colors, 0, 0, cardinality8, scale, shape, cardinality4_mapping[corner], bg=0, inter_shape_spacing=0, scale=line_length)      
            working_board2 = remove_padding(working_board2, 0)
        elif riddle_type == "scale_f_line_length_rotate_starting_f_corner_plus_90_deg_per_repeat_color_last_and_first_in_fill_direction_f_line_direction_inter_item_distance_of_0_in_fill_f_1st_line_color_random_line_colors":
            # set all elements in line_colors to the value in the last element
            shape_colors = np.full(scale, line_colors[-1])
            #draw_repeating_shapes_in_line_f_point(board, bring_to_front: bool, colors: list, startx: int, starty: int, line_cardinality8: int, length: int, shape: str, shape_cardinality_start=-1, bg=0, inter_shape_spacing=1, scale=1):
            working_board2=draw_repeating_shapes_in_line_f_point(working_board2, True, shape_colors, 0, 0, cardinality8, scale, shape, cardinality4_mapping[corner], bg=0, inter_shape_spacing=0, scale=line_length, in_fill=line_colors[0], in_fill_function=shape_color_and_in_fill)
            working_board2 = remove_padding(working_board2, 0)
        elif riddle_type == "scale_f_line_length_rotate_starting_f_corner_plus_90_deg_per_repeat_color_last_and_first_in_fill_direction_f_line_direction_inter_item_distance_of_0_in_fill_joined_f_1st_line_color_random_line_colors":
            # set all elements in line_colors to the value in the last element
            shape_colors = np.full(scale, line_colors[-1])
            #draw_repeating_shapes_in_line_f_point(board, bring_to_front: bool, colors: list, startx: int, starty: int, line_cardinality8: int, length: int, shape: str, shape_cardinality_start=-1, bg=0, inter_shape_spacing=1, scale=1):
            working_board2=draw_repeating_shapes_in_line_f_point(working_board2, True, shape_colors, 0, 0, cardinality8, scale, shape, cardinality4_mapping[corner], bg=0, inter_shape_spacing=0, scale=line_length)
            working_board2 = shape_color_and_in_fill(remove_padding(working_board2, 0), shape_colors[0], line_colors[0])
        elif riddle_type == "scale_f_line_length_rotate_starting_f_corner_plus_90_deg_per_repeat_color_last_and_first_in_fill_direction_f_line_direction_inter_item_distance_of_0_checkered_in_fill_joined_f_1st_line_color_random_line_colors":
            # set all elements in line_colors to the value in the last element
            shape_colors = np.full(scale, line_colors[-1])
            #draw_repeating_shapes_in_line_f_point(board, bring_to_front: bool, colors: list, startx: int, starty: int, line_cardinality8: int, length: int, shape: str, shape_cardinality_start=-1, bg=0, inter_shape_spacing=1, scale=1):
            working_board2=draw_repeating_shapes_in_line_f_point(working_board2, True, shape_colors, 0, 0, cardinality8, scale, shape, cardinality4_mapping[corner], bg=0, inter_shape_spacing=0, scale=line_length)
            working_board2 = in_fill_shape_checkerboard(remove_padding(working_board2, 0), shape_colors[0], line_colors[0])   
        elif riddle_type == "padding_f_scale_shape_repeat_f_line_length_dir_f_line_dir_rotate_starting_with_cardinality_f_corner_plus_90_deg_per_shape_repeat_shape_color_f_line_shape_scale_same_inter_shape_distance_f_scale_random_line_colors_in_fill_f_shape_color":
            working_board2=draw_repeating_shapes_in_line_f_point(working_board2, True, line_colors, 0, 0, cardinality8, line_length, shape, cardinality4_mapping[corner], bg=0, inter_shape_spacing=scale, scale=line_length, in_fill=color_shape, in_fill_function=shape_color_and_in_fill)
            working_board2=add_padding(working_board2, scale, color_shape)
#cardinality4_mapping[corner]
        if reverse_input_output:
            input_grids.append(working_board2)
            output_grids.append(working_board)
        else:
            input_grids.append(working_board)
            output_grids.append(working_board2)
        params['input'].append({'shape': shape.__name__, 'scale': scale, 'color': color_shape, 'padding': scale, 'corner': corner, 'line_length': line_length, 'line_color': color_line, "cardinality8": cardinality8})
        params['output'].append({'shape': shape.__name__, 'scale': line_length, 'color': color_line, 'padding': 0, 'corner': corner, 'line_length': line_length, 'line_color': color_line})
    return input_grids, output_grids, params    