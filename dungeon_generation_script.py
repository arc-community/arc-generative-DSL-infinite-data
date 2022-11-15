


import matplotlib.pyplot as plt
from colored import attr, bg, fg
from matplotlib.colors import ListedColormap

TRANS_PIXEL = -1
OOB_PIXEL = -2

CELL_PADDING_STR=" " * 1
#BOARD_GAP_STR = " " * 6
BOARD_BEG_STR =           "|" + " " * 3  #string at beginning of board
BOARD_GAP_STR = " " * 3 + "|" + " " * 3 
PAIR_GAP_STR = "\n" + ("~" * 32) + "\n"
#"xd" #\n" + " " * settings.pair_gap + "\n"

colors=["red", "orange", "gold", "limegreen", "k", 
        "#550011", "purple", "seagreen"]
colors=["black","blue","red","limegreen","gold",
        "gray","purple","darkorange", "cyan", "maroon",]
plt_cmap = ListedColormap(colors)
colors2=["lightyellow","black","blue","red","limegreen","gold",
        "gray","purple","darkorange", "cyan", "maroon","lightyellow"]
plt_cmap2 = ListedColormap(colors2)

def nbhd8(X,r,c): #just return values
    nbhd8_inds = [(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1)]
    h,w=X.shape

    nbhd_list=[]
    for ind in nbhd8_inds:
        r2=r+ind[0]
        c2=c+ind[1]
        if (c2>=0 and c2<h) and (r2>=0 and r2<w):
            nbhd_list.append( X[r2,c2] )
        else:
            nbhd_list.append( OOB_PIXEL )
            
    return nbhd_list

def nbhd9(X,r,c): #just return values
    nbhd9_inds = [(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1),  (0,0),]
    h,w=X.shape

    nbhd_list=[]
    for ind in nbhd9_inds:
        r2=r+ind[0]
        c2=c+ind[1]
        if (c2>=0 and c2<w) and (r2>=0 and r2<h):
            nbhd_list.append( X[r2,c2] )
        else:
            nbhd_list.append( OOB_PIXEL )
            
    return nbhd_list

def replace_color(X,color1,color2):
    X=np.copy(X)
    X[X==color1] = color2
    return X



#apply a cellular automata rule
#for now only assuming 9-neighborhood
def apply_CA_rule(X,rule):
    Y=np.zeros_like(X,dtype=int)

    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            #nbhd_list = 
            nbhd_list=nbhd9(X,r,c)
            Y[r,c] = rule(nbhd_list)
            
    return Y






class agent_09c534e7_unsolve():
    def unsolve(self,Y):

        def local_CA_rule(nbhd_list):
            if 0 in nbhd_list:
                if nbhd_list[8]==0:
                    return 0
                else:
                    return 6
            else:
                return nbhd_list[8]


            
        X = apply_CA_rule(Y, local_CA_rule)
        return X




import numpy as np

#seed=69
#np.random.seed(seed)
#perm = np.random.permutation(400)
#print(perm[:10])


super_verbose = True
super_verbose = False

#~~~ Resources ~~~
#-http://www.roguebasin.com/index.php/Basic_BSP_Dungeon_generation
#-https://en.wikipedia.org/wiki/Binary_space_partitioning
#-https://graphics.tudelft.nl/Publications-new/2016/SLTLB16/chapter03.Online.pdf
#~~~~~~~~~~~~~~~~~
def generate_09c534e7(room_border_color=4): #purple dungeon with several connected components
    components = np.random.randint(2)+2 #2 to 3
    C=components
    #C=1
    #C=2
    if super_verbose:
        print('digger_total',C)

    #min_w = 10;
    #max_w = 25;
    #min_h = 10;
    #max_h = 25;
    min_w = 20;
    max_w = 31;
    min_h = 20;
    max_h = 31;

    room_min_w = 3;
    #room_max_w = 10;
    room_max_w = 8;
    room_min_h = 3;
    #room_max_h = 10;
    room_max_h = 8;

    h = np.random.randint(min_h,max_h)
    w = np.random.randint(min_w,max_w)
    #h,w=20,20
    #h,w=30,30

    board = np.zeros( (h,w),dtype=int)
    #max_placement_tries = 10
    max_placement_tries = 20
    max_room_placement_tries = 10 #todo make more efficient only search over valid space
    
    #TODO 50 minutes invested so far
    vec = [[0,1],[-1,0],[0,-1],[1,0]]
    T = np.random.randint(3,10)
    #T=10
    #T=2
    #T=3
    locs = np.zeros((T,C,2),dtype=int)


    
    
    xs=[];ys=[];
    for c in range(C):
        tries = 0
        found = False
        tries2 = 0
        while tries<max_placement_tries:
            y = np.random.randint(2,h-2)
            x = np.random.randint(2,w-2)
            #print("C",c,"\t",y,x)
            if not helper_check_room(board,y-2,x-2,y+5-2,x+5-2):
                tries+=1
                #print('t1',tries)
            else:
                tries=max_placement_tries
                found=True
                #print(y,x)
                #print('found center')
                
                
                tries2 = 0
                found2 = False
                #while tries2<max_room_placement_tries:
                while not found2: 
                    rh = np.random.randint(room_min_h,room_max_h)
                    rw = np.random.randint(room_min_w,room_max_w)
                    ry = np.random.randint(1,rh-1)
                    rx = np.random.randint(1,rw-1)
                    #print('t2',tries2,'\troom',y,x,rh,rw,ry,rx)
                    #print('\t\t',y-ry,x-rx,y-ry+rh,x-rx+rw)

                    if not helper_check_room(board,y-ry,x-rx,y-ry+rh,x-rx+rw):
                        tries2 += 1
                        #if tries2>5:
                        #    board[y,x] = c+1+2
                        #    plt.imshow(board,cmap=plt_cmap2);plt.clim(-1.5,10.5);plt.colorbar();plt.show();
                        #    board[y,x] = 0
                    else:
                        #place room
                        ###placed = helper_place(board,y-ry,x-rx,y-ry+rh,x-rx+rw,6+c)
                        #placed = helper_place(board,y-ry,x-rx,y-ry+rh,x-rx+rw)
                        placed = helper_place(board,y-ry,x-rx,y-ry+rh,x-rx+rw,c+1)
                        #place new location
                        board[y,x] = c+1
                        locs[0,c] = [y,x]
                        found2 = True
                #print(tries2)
                tries=max_placement_tries
                found=True
                #print()
                #print()
        
    if super_verbose:
        print("YEP")

    
    for t in range(1,T):
        #plt.imshow(board,cmap=plt_cmap2);plt.clim(-1.5,10.5);plt.colorbar();plt.show();
        for c in range(C):
            tries = 0
            found = False
            while tries<max_placement_tries:
                s = np.random.randint(t)
                xy = locs[s,c]
                y=xy[0]
                x=xy[1]

                direc_int = np.random.randint(4)
                #direc = vec[direc_int]
                length = np.random.randint(3,8)
                #print('direc',direc_int,'length',length)

                y1=y;y2=y+1;x1=x;x2=x+1;
                xx=x;yy=y;
                if direc_int==0:
                    y2=y+length+1
                    yy=y+length
                elif direc_int==1:
                    x1=x-length
                    xx=x-length
                elif direc_int==2:
                    y1=y-length
                    yy=y-length
                elif direc_int==3:
                    x2=x+length+1
                    xx=x+length
                    
                #if not helper_check_room(board,yy-3,xx-3,yy-3+1,xx-3+1):
                if not helper_check_room(board,yy-1,xx-1,yy-1+3,xx-1+3):
                    tries += 1
                else:
                    tries2=0
                    while tries2 < max_room_placement_tries:
                        rh = np.random.randint(room_min_h,room_max_h)
                        rw = np.random.randint(room_min_w,room_max_w)
                        ry = np.random.randint(1,rh-1)
                        rx = np.random.randint(1,rw-1)
                    
                        if not helper_check_room(board,yy-ry,xx-rx,yy-ry+rh,xx-rx+rw):
                            tries2 += 1
                        else:
                            #place room
                            ###placed = helper_place(board,yy-ry,xx-rx,yy-ry+rh,xx-rx+rw,9+t-1)
                            #placed = helper_place(board,yy-ry,xx-rx,yy-ry+rh,xx-rx+rw)
                            placed = helper_place(board,yy-ry,xx-rx,yy-ry+rh,xx-rx+rw,c+1)
                            #place hallway
                            ###placed = helper_place(board,y1,x1,y2,x2,8)
                            placed = helper_place(board,y1,x1,y2,x2,c+1)
                            #place new location
                            ###board[yy,xx] = c+5

                            #sampling new yy,xx so not as uniform hallway alignment
                            yy=np.random.randint(yy-ry+1,yy-ry+rh-1)
                            xx=np.random.randint(xx-rx+1,xx-rx+rw-1)
                            board[yy,xx] = c+1
                            locs[t,c] = [yy,xx]
                            tries2 = max_room_placement_tries
                            tries = max_placement_tries
                            found = True


                            
            if not found:
                if super_verbose:
                    print("GIVING UP at",t)
                locs[t,c] = [-1,-1]
            

    #plt.imshow(board,cmap=plt_cmap2);plt.clim(-1.5,10.5);plt.colorbar();plt.show();

    #recoloring how I feel (out of 5 colors in four true input boards in ARC)
    my_colors = []
    board=-board
    for c in range(C):
        while len(my_colors)<(c+1):
            col = np.random.randint(1,6)
            if col==5:
                col=8
            if col not in my_colors:
                my_colors.append(col)
                board[board==-(c+1)] = col
    if super_verbose:
        print('my colors',my_colors)
    #plt.imshow(board,cmap=plt_cmap2);plt.clim(-1.5,10.5);plt.colorbar();plt.show();
    
    inp_board = np.copy(board)
    out_board = np.copy(board)

    inp_board[inp_board!=0] = 6
    my_starts = []
    for c in range(C):
        while len(my_starts)<(c+1):
            s = np.random.randint(T)
            xy = locs[s,c]
            y=xy[0]
            x=xy[1]
            if x!=-1 and y!=-1:
                inp_board[y,x] = my_colors[c]
                my_starts.append(True)
    #plt.imshow(inp_board,cmap=plt_cmap2);plt.clim(-1.5,10.5);plt.colorbar();plt.show();

    out_board = purplify_borders(out_board)
    #xd = agent_09c534e7_unsolve(out_board)
    #plt.imshow(out_board,cmap=plt_cmap2);plt.clim(-1.5,10.5);plt.colorbar();plt.show();

    return inp_board,out_board

def purplify_borders(out_board):
    agent = agent_09c534e7_unsolve()
    board2 = agent.unsolve(out_board)
    return board2
    
def helper_place(board,y1,x1,y2,x2,cc=6):
    if super_verbose:
        print('helper',y1,x1,y2,x2)
    y1=int(np.maximum(y1,0));y2=int(np.minimum(y2,board.shape[0]));
    x1=int(np.maximum(x1,0));x2=int(np.minimum(x2,board.shape[1]));
    if super_verbose:
        print('helper',y1,x1,y2,x2)
    new_board = np.array(board)
    new_board = (board)
    new_board[y1:y2,x1:x2] = cc


def helper_check(board,y1,x1,y2,x2):
    valid_onboard = (y1>=0) and (x1>=0) and (y2<board.shape[0]) and (x2<board.shape[1])

    percent_nonOOB = 0.0
    sub_board = board[y1:y2,x1:x2]
    #print('sub size ',sub_board.size)
    #print('sub shape',sub_board.shape)
    if sub_board.size>0:
        percent_nonOOB = sub_board.shape[0]*sub_board.shape[1]/(x2-x1)/(y2-y1)
    #print('non_OOB',percent_nonOOB)
    
    percent_nonzero = 1.0
    if (percent_nonOOB>0.0):
        percent_nonzero = np.mean( (sub_board!=0).astype(int) )
    valid_nonpurple = (percent_nonzero==0.0)
    #print('nonzero',percent_nonzero)
    
    ###valid_onboard   =  (percent_nonOOB==1.0)
    ###return (valid_nonpurple and valid_onboard)
    return valid_nonpurple,valid_onboard

def helper_check_room(board,y1,x1,y2,x2):
    valid_nonpurple5,valid_onboard5 = helper_check(board,y1-1,x1-1,y2+1,x2+1)
    valid_nonpurple3,valid_onboard3 = helper_check(board,y1,x1,y2,x2)
    valid = (valid_nonpurple5 and valid_onboard3 and valid_nonpurple3)
    return valid




def generate_full_riddle_09c534e7():
    inp_board,out_board = generate_09c534e7()
    plt.imshow(inp_board,cmap=plt_cmap2);plt.clim(-1.5,10.5);plt.colorbar();plt.show();
    plt.imshow(out_board,cmap=plt_cmap2);plt.clim(-1.5,10.5);plt.colorbar();plt.show();
    return inp_board,out_board




if __name__ == "__main__":
    inp_board,out_board = generate_09c534e7()
    plt.clf();
    plt.imshow(inp_board,cmap=plt_cmap2);plt.clim(-1.5,10.5);plt.colorbar();
    plt.show();
    plt.clf();
    plt.imshow(out_board,cmap=plt_cmap2);plt.clim(-1.5,10.5);plt.colorbar();
    plt.show();



