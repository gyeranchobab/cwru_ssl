import matplotlib.pyplot as plt

def draw_plot(tsne, label, cmap = None, legend = False, ax=None):
    if ax is None:
        ax = plt.subplot(1,1,1)
    ax.set_xticks([])
    ax.set_yticks([])
    if cmap is None:
        cmap = plt.cm.get_cmap('coolwarm')
    if not legend:
        ax.scatter(tsne[:,0], tsne[:,1], s=1, cmap=cmap, c=label, picker=5)
    else:
        set_l = set(label)
        for i, l in enumerate(set_l):
            ax.scatter(tsne[:,0][label==l], tsne[:,1][label==l], c=cmap(i/len(set_l)), cmap=cmap, label=l, s=1, picker=5)
            ax.legend()
    plt.show()
    
def draw_magic(tsne, label, raw, index, cmap = None, frame_len=1024, frame_intv=512):
    global p, m,pick
    p=None
    fig=plt.figure(figsize=(8,8))
    ax0 = plt.subplot2grid((4,1),(0,0),rowspan=3)#(3,1,1)
    ax1 = plt.subplot(414)
    
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    m=0
    pick=False
    text = ax0.text(0,100,"Label %d"%m,va='top',ha='left')
    def onpress(event):
        global m,pick
        if event.key=='right' and m < len(label)-1:
            m+=1
        elif event.key == 'left' and m > 0:
            m-=1
        else:
            return
        if pick:
            p.remove()
            pick=False
        ax0.cla()
        ax0.set_xticks([])
        ax0.set_yticks([])
        text = ax0.text(0,100,"Label %d"%m,va='top',ha='left')
        draw_plot(tsne, label[m], cmap, legend=False, ax=ax0)
        fig.canvas.draw()
        
    def onpick(event):
        global p,pick
        pt = event.ind[0]
        if pick:
            p.remove()
            pick=False
        p = ax0.scatter(tsne[pt,0],tsne[pt,1],marker='x',s=10,c='k')
        pick=True
        
        ax1.cla()
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.plot(raw[index[pt]:index[pt]+frame_len][:,0])
        fig.canvas.draw()
        
    cid = fig.canvas.mpl_connect('pick_event', onpick)
    cid2 = fig.canvas.mpl_connect('key_press_event', onpress)
    draw_plot(tsne, label[m], cmap, legend=False, ax=ax0)
    plt.show()
    