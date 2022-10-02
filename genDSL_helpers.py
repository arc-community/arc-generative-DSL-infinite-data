


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
colors2=["lightyellow","black","blue","red","limegreen","gold",
        "gray","purple","darkorange", "cyan", "maroon","lightyellow"]
plt_cmap2 = ListedColormap(colors2)
def visualize_board(board, gridlines=False):
    fig_y = np.maximum(5, board.shape[0]/6)
    fig_x = np.maximum(7, board.shape[1]/6+1)
    plt.figure(figsize=(fig_x,fig_y))
    plt.imshow(board,cmap=plt_cmap2);
    if gridlines:
        yy=np.arange(board.shape[0])+0.5;xx=np.arange(board.shape[1])+0.5;
        plt.hlines(y=yy,xmin=-0.5,xmax=board.shape[1]-0.5, color='darkgray')
        plt.vlines(x=xx,ymin=-0.5,ymax=board.shape[0]-0.5, color='darkgray')
    plt.clim(-1.5,10.5);plt.colorbar();plt.show();
