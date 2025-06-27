import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import cbook
from mpl_toolkits.axes_grid1 import ImageGrid

imgs = np.load('imagenet_ditb_imgs.npy')
labs = np.load('imagenet_ditb_labs.npy')
print(imgs.shape)
print(labs.shape)
imgs = imgs * 0.5 + 0.5
imgs = np.clip(imgs, 0, 1)

fontLegend = 16
fontAxis = 16
fontText = 16

b = 2
for b in range(64):
    plt.figure()
    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'figure.dpi': 300})
    plt.rcParams.update({'savefig.dpi': 300})
    plt.rcParams.update({'mathtext.fontset': 'cm'})
    print(b)
    fig = plt.figure(figsize=(30, 30))
    # A grid of 3x3 images with a single colorbar.
    grid = ImageGrid(fig, 111, nrows_ncols=(3, 3), axes_pad=0.0, label_mode="L", share_all=True)
    for i,ax in enumerate(grid):
        im = ax.imshow(imgs[i,b]) #, extent=extent)
        ax.axis('off')

    fig.patch.set_visible(False)
    #plt.show()
    name = 'images_imagenet_ditb/qual_{}_{}'.format(b, labs[0,b])
    plt.savefig(name + '.png', bbox_inches='tight')
    #plt.savefig(name + '.svg', format="svg", bbox_inches='tight')
    pdf = PdfPages(name + '.pdf'.format(b))
    pdf.savefig(bbox_inches='tight')
    pdf.close()
    plt.close(fig)
    print('Plots done!')