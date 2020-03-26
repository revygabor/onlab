import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dir = '20200325194037_new_deep_lab_v3_plus_citryscapes_fine'


def plot_iou_barh(labels, iou_vals):
	fig, ax = plt.subplots(figsize=(7, 5))
	pd.DataFrame({'iou value': iou_vals}, index=labels).plot.barh(ax=ax)
	# set individual bar lables using above list
	for i in ax.patches:
		# get_width pulls left or right; get_y pushes up or down
		ax.text(i.get_width()+.005, i.get_y()+.05, \
	            str(i.get_width()),
				color='dimgrey')
	plt.xlim(0, 1.05)
	plt.tight_layout()

	return fig, ax



def plot_confusion_matrix(conf_mtx, classes,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    if not title:
        title = 'Normalized confusion matrix'

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(conf_mtx, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(conf_mtx.shape[1]),
           yticks=np.arange(conf_mtx.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = conf_mtx.max() / 2.
    for i in range(conf_mtx.shape[0]):
        for j in range(conf_mtx.shape[1]):
            ax.text(j, i, format(conf_mtx[i, j], fmt),
                    ha="center", va="center",
                    color="white" if conf_mtx[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, ax

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('dir', metavar='dir', type=str,
		help='an integer for the accumulator')
	args = parser.parse_args()
	dir = args.dir

	# iou
	iou_csv_file_path = glob('{}/*.csv'.format(dir))[0]
	iou_csv_file = pd.read_csv(iou_csv_file_path, header=None, sep='\t', names=['iou value'], index_col=0)
	iou_vals = iou_csv_file['iou value'].to_list()
	labels = iou_csv_file.index.to_list()
	fig, ax = plot_iou_barh(iou_vals=iou_vals, labels=labels)
	# plt.show(block=True)
	fig.savefig('{}.svg'.format(iou_csv_file_path), pad_inches=0, format='svg')



	# conf mtx
	conf_mtx_file_path = glob('{}/*.npy'.format(dir))[0]
	conf_mtx = np.load(conf_mtx_file_path)
	fig, ax = plot_confusion_matrix(conf_mtx, classes=labels, title=" ", cmap=plt.cm.Reds)
	# plt.show(block=True)
	fig.savefig('{}.svg'.format(conf_mtx_file_path), pad_inches=0, format='svg')

