


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
# build a color map of these colors "#000000", "#0074D9", "#FF4136", "#2ECC40","#FFDC00","#AAAAAA","#F012BE","#FF851B","#7FDBFF","#870C25",
cmap_list = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
# convert the hex strings to values compatible with matplotlib
cmap_values = ListedColormap(cmap_list)
norm = Normalize(vmin=0, vmax=9)

def visualize_board(board, gridlines=False):
    fig_y = np.maximum(5, board.shape[0]/6)
    fig_x = np.maximum(7, board.shape[1]/6+1)
    plt.figure(figsize=(fig_x,fig_y))
    plt.imshow(board,cmap=cmap_values, norm=norm, origin='lower');
    if gridlines:
        yy=np.arange(board.shape[0])+0.5;xx=np.arange(board.shape[1])+0.5;
        plt.hlines(y=yy,xmin=-0.5,xmax=board.shape[1]-0.5, color='darkgray')
        plt.vlines(x=xx,ymin=-0.5,ymax=board.shape[0]-0.5, color='darkgray')
    plt.clim(-1.5,10.5);plt.colorbar();plt.show();


def plot_riddle(train_input_grids, train_output_grids, test_input_grids, test_output_grids):
    # plot the grids next to each other input in the first row and output in the second row
    fig, axs = plt.subplots(2, len(train_input_grids))
    title = f"train input"
    axs[0][0].set_title(title)
    title2 = f"train output"
    axs[1][0].set_title(title2)
    for r in range(len(train_input_grids)):
        axs[0][r].axis('off')
        axs[1][r].axis('off')
        axs[0][r].imshow(train_input_grids[r], cmap=cmap_values, norm=norm, origin='lower')
        axs[1][r].imshow(train_output_grids[r], cmap=cmap_values, norm=norm, origin='lower')
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

