
import cv2
import PIL
from PIL import ImageDraw, ImageFont
from matplotlib import patches, patheffects
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.ticker as plticker
import torch
import numpy as np
import itertools


# PLT functions

def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])
    
def draw_rect(ax, b):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor='white', lw=0.5))
    draw_outline(patch, 2)
    
def draw_text(ax, xy, txt, sz=14):
    text = ax.text(*xy, txt,
        verticalalignment='top', color='white', fontsize=sz, weight='bold')
    draw_outline(text, 1)


def show_grid(ax, grid_interval):
    loc = plticker.MultipleLocator(base=grid_interval)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    ax.grid(which='major', axis='both', linestyle='-')
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    return ax

def show_img(im, figsize=None, ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax

def draw_axis(ax, isDraw=False):
    ax.get_xaxis().set_visible(isDraw)
    ax.get_yaxis().set_visible(isDraw)

def draw_image(ax, img):
    ax.imshow(img, shape=(2,2), aspect='equal')

def draw_labels(ax, labels, classes, coord_idx=[1,2,3,4], class_idx=0):
    for l in labels:
        if l.sum() == 0:
            continue
        rect = l[coord_idx]
        c = classes[l[class_idx].astype(np.int32)]
        draw_rect(ax, rect)
        draw_text(ax, rect[:2], c)
        

# Display a list of images with labels in given grid size         
def show_img_grid(img_list, classes=None,
                  labels_list=None, coord_idx=[1,2,3,4], class_idx=0,
                  cols=2, figsize=None, col_title_dict=None):
    rows = int(np.ceil(len(img_list) / cols))

    heights = [a.shape[0] for a in img_list[::cols]]
    widths = [a.shape[1] for a in img_list[0:cols]]

    fig_width = 25  # inches
    fig_height = fig_width * sum(heights) / sum(widths)
    fig_size = (fig_width, fig_height)

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=fig_size, gridspec_kw={'height_ratios':heights})
    axes = [ax for ax in axes.ravel()]

    if col_title_dict is not None:
        assert(cols == len(col_title_dict['title']))
        for ax, col in zip(axes[:cols], col_title_dict['title']):
            ax.set_title(col, pad=col_title_dict['pad'], 
                         fontdict={'fontsize': col_title_dict['fontsize'],
                                   'fontweight' : col_title_dict['fontweight'] })
    
    if labels_list is None:
        labels_list = []

    for ax, img, labels in itertools.zip_longest(axes, img_list, labels_list, fillvalue=None):
        if img is not None:
            draw_image(ax, img)
        if labels is not None:
            draw_labels(ax, labels, classes, coord_idx=coord_idx, class_idx=class_idx)
        draw_axis(ax, False)
        
    plt.subplots_adjust(wspace=0.01, hspace=0.02, left=0, right=1, bottom=0, top=1)
    plt.tight_layout()

# Combine images to one image
def getImgGrid(img_list, cols=2):
    num_img, height, width, channel = img_list.shape
    rows = num_img // cols
    # target = (height * rows, width * cols, channel)
    grid = img_list.reshape(rows, cols, height, width, channel)\
                    .transpose(0, 2, 1, 3, 4)\
                    .reshape(height * rows, width * cols, channel)
    return grid

def get_color_pallete(num_color):
    cmap = plt.get_cmap('tab20b')
    colors = torch.Tensor([cmap(i) for i in np.linspace(0, 1, num_color)])
    bbox_colors = colors[torch.randperm(num_color)]
    return bbox_colors

 # openCV functions

def cv2_drawTextWithBkgd(img, text, bt_left_pt, color, max_x, max_y, font_scale=2.0, font=cv2.FONT_HERSHEY_PLAIN):
    (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, fontScale=font_scale, thickness=1)[0]
        
    t_pt1 = np.clip(bt_left_pt[0], 0, max_x - text_width), np.clip(bt_left_pt[1], text_height, max_y) 
    t_pt2 = t_pt1[0] + text_width, t_pt1[1] - text_height
    
    img = cv2.rectangle(img, t_pt1, t_pt2, color, cv2.FILLED, 4)
    img = cv2.putText(img, text, t_pt1, cv2.FONT_HERSHEY_PLAIN, fontScale=font_scale, color=(0, 0, 0), thickness=2);
    return img

