


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import colors
# build a color map of these colors "#000000", "#0074D9", "#FF4136", "#2ECC40","#FFDC00","#AAAAAA","#F012BE","#FF851B","#7FDBFF","#870C25",
cmap_list = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
# convert the hex strings to values compatible with matplotlib
cmap_values = ListedColormap(cmap_list)
norm = Normalize(vmin=0, vmax=9)

def visualize_board(board, gridlines=False):
    # plot a board
    fig, ax = plt.subplots(1, 1)
    ax.axis('off')
    plt.imshow(board, cmap=cmap_values, norm=norm)
    if gridlines:
        yy=np.arange(board.shape[0])+0.5;xx=np.arange(board.shape[1])+0.5;
        plt.hlines(y=yy,xmin=-0.5,xmax=board.shape[1]-0.5, color='gray')
        plt.vlines(x=xx,ymin=-0.5,ymax=board.shape[0]-0.5, color='gray')
    plt.clim(0,10);  plt.colorbar(); plt.show()
    
    
    




    
    # fig_y = np.maximum(5, board.shape[0]/6)
    # fig_x = np.maximum(7, board.shape[1]/6+1)
    # plt.figure(figsize=(fig_x,fig_y))
    # plt.imshow(board,cmap=cmap_values, norm=norm, origin='lower');
    # if gridlines:
    #     yy=np.arange(board.shape[0])+0.5;xx=np.arange(board.shape[1])+0.5;
    #     plt.hlines(y=yy,xmin=-0.5,xmax=board.shape[1]-0.5, color='darkgray')
    #     plt.vlines(x=xx,ymin=-0.5,ymax=board.shape[0]-0.5, color='darkgray')
    # plt.clim(-1.5,10.5);plt.colorbar();plt.show();


def plot_riddle(train_input_grids, train_output_grids, test_input_grids, test_output_grids, gridlines=False):
    # plot the grids next to each other input in the first row and output in the second row
    fig, axs = plt.subplots(2, len(train_input_grids))
    title = f"train input"
    axs[0][0].set_title(title)
    title2 = f"train output"
    axs[1][0].set_title(title2)
    line_width = 0.5
    for r in range(len(train_input_grids)):
        axs[0][r].axis('off')
        axs[1][r].axis('off')
        axs[0][r].imshow(train_input_grids[r], cmap=cmap_values, norm=norm, origin='lower')
        axs[1][r].imshow(train_output_grids[r], cmap=cmap_values, norm=norm, origin='lower')
        if gridlines:
            yy=np.arange(train_input_grids[r].shape[0])+0.5;xx=np.arange(train_input_grids[r].shape[1])+0.5;
            axs[0][r].hlines(y=yy,xmin=-0.5,xmax=train_input_grids[r].shape[1]-0.5, color='gray', linewidth=line_width)
            axs[0][r].vlines(x=xx,ymin=-0.5,ymax=train_input_grids[r].shape[0]-0.5, color='gray', linewidth=line_width)
            yy=np.arange(train_output_grids[r].shape[0])+0.5;xx=np.arange(train_output_grids[r].shape[1])+0.5;
            axs[1][r].hlines(y=yy,xmin=-0.5,xmax=train_output_grids[r].shape[1]-0.5, color='gray', linewidth=line_width)
            axs[1][r].vlines(x=xx,ymin=-0.5,ymax=train_output_grids[r].shape[0]-0.5, color='gray', linewidth=line_width)
            # rewrite the above to use a more standard method of drawing gridlines



    plt.show()    
    # plot test items
    fig, axs = plt.subplots(2, 1)
    title = f"test input"
    axs[0].set_title(title)
    axs[0].axis('off')
    title2 = f"test output"
    axs[1].set_title(title2)
    axs[1].axis('off')
    axs[0].imshow(test_input_grids[0], label="test input", cmap=cmap_values, norm=norm, origin='lower')
    axs[1].imshow(test_output_grids[0], label="test output", cmap=cmap_values, norm=norm, origin='lower')


def plot_one(ax, train_or_test, input_or_output, input_matrix):
    cmap = colors.ListedColormap(
        [
            "#000000",
            "#0074D9",
            "#FF4136",
            "#2ECC40",
            "#FFDC00",
            "#AAAAAA",
            "#F012BE",
            "#FF851B",
            "#7FDBFF",
            "#870C25",
        ]
    )
    norm = colors.Normalize(vmin=0, vmax=9)

    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which="both", color="lightgrey", linewidth=0.5)
    ax.set_yticks([x - 0.5 for x in range(1 + len(input_matrix))])
    ax.set_xticks([x - 0.5 for x in range(1 + len(input_matrix[0]))])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(train_or_test + " " + input_or_output)

def plot_task(train_input_grids, train_output_grids, test_input_grids, test_output_grids, gridlines=False):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    num_train = len(train_input_grids)
    fig, axs = plt.subplots(2, num_train, figsize=(3 * num_train, 3 * 2))
    for i in range(num_train):
        plot_one(axs[0, i], "train", "input", train_input_grids[i])
        plot_one(axs[1, i], "train", "output", train_output_grids[i])
    plt.tight_layout()
    plt.show(block=False)

    num_test = len(test_input_grids)
    fig, axs = plt.subplots(num_test, 2, figsize=(3 * 2, 3 * num_test))
    if num_test == 1:
        plot_one(axs[0], "test", "input", test_input_grids[0])
        plot_one(axs[1], "test", "output", test_output_grids[0])
    else:
        for i in range(num_test):
            plot_one(axs[0, i], "test", "input", test_input_grids[i])
            plot_one(axs[1, i], "test", "output", test_output_grids[i])
    plt.tight_layout()
    plt.show(block=False)

def plot_board(board, gridlines=True):
    ax = None
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(board, cmap=cmap_values, norm=norm)
    if gridlines:
        ax.grid(True, which="both", color="lightgrey", linewidth=0.5)
        ax.grid(True, which="both", color="lightgrey", linewidth=0.5)
        ax.set_yticks([y - 0.5 for y in range(board.shape[0])])
        ax.set_xticks([x - 0.5 for x in range(board.shape[1])])
        ax.set_xticklabels([])
        ax.set_yticklabels([])        
    #ax.axis('off')
