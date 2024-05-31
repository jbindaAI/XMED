from scipy import ndimage
import seaborn as sns
import torchio as tio
import os
import numpy as np
from typing import List, Tuple
import io
import base64
import torch
from models.att_cdam_utils import get_cmap
import matplotlib.pyplot as plt
import pylidc as pl
from pylidc.utils import consensus
import pickle


# Loading annotations for all dataset
with open("data/ALL_annotations_df.pkl", "rb") as file:
    ann_df = pickle.load(file)


def plot_res_class(maps):
    """Using matplotlib, plot the original image and the relevance maps"""
    maps2plot=[]
    target_names=[]
    maps2plot.append(maps[0])
    target_names.append("Attention Map")
    for key in maps[1].keys():
        maps2plot.append(maps[1][key])
        target_names.append("CDAM score")

    if len(maps2plot) == 2:
        # Binary classification:
        plt.figure(figsize=(6,3))
        num_plots = 2
        for i, m in enumerate(maps2plot):
            plt.subplot(1, num_plots, i + 1, title=target_names[i])
            plt.imshow(m, cmap=get_cmap(m))
            plt.axis("off")
        plt.subplots_adjust(wspace=0.005, hspace=0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", transparent=True, bbox_inches='tight')
    buf.seek(0)
    res_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return res_str


def plot_CDAM_reg(maps, preds):
    """Using matplotlib, plot the original image and the relevance maps"""
    maps2plot=[]
    target_names=[]
    for key in maps[0].keys():
        maps2plot.append(maps[0][key])
        target_names.append(key)

    # Biomarker Regression:
    fig, axs = plt.subplots(4, 4, figsize=(10, 10))

    # Plotting CDAM scores:
    k = 0
    for i in [0, 2]:
        for j in range(4):
            axs[i][j].imshow(maps2plot[k], cmap=get_cmap(maps2plot[k]))
            axs[i][j].set_title(target_names[k])
            axs[i][j].tick_params(axis='both', 
                                  which='both', 
                                  bottom=False, 
                                  left=False,
                                  labelbottom=False,
                                  labelleft=False
                                  )
            k += 1
    k = 0
    for i in [1, 3]:
        for j in range(4):
            sns.histplot(ann_df,
                         x=target_names[k].lower(),
                         kde=True,
                         bins=16,
                         stat="percent",
                         ax=axs[i][j])
            axs[i][j].axvline(x=preds[target_names[k]],
                              color='red',
                              linestyle='--',
                              linewidth=2,
                              label=r'$\hat{y}$'
                              )
            axs[i][j].set_title(target_names[k] + "=" + str(preds[target_names[k]]))
            axs[i][j].legend()
            k += 1
    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", transparent=True, bbox_inches='tight')
    buf.seek(0)

    res_str_CDAM = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return res_str_CDAM


def plot_att_reg(attention_map):
    plt.figure(figsize=(3.3,3.3))
    plt.imshow(attention_map, cmap=get_cmap(attention_map))
    plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", transparent=True)
    buf.seek(0)

    res_str_att_reg = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return res_str_att_reg


## Tutaj należałoby trochę poprawić, by lepiej współpracowało z process_nodule
def plot_nodule(original_img:torch.Tensor)->str:
    plt.figure(figsize=(3.3,3.3))
    plt.set_cmap('gray')
    plt.imshow(original_img[:, :, 0])
    plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", transparent=True)
    buf.seek(0)

    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_str


def load_dicom_into_tensor(patient_id: str)->torch.Tensor:
    if "dt_" + patient_id+".pt" not in os.listdir("cache/"):
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == f"LIDC-IDRI-{patient_id}").first()
        dicom = scan.get_path_to_dicom_files()
        tio_image = tio.ScalarImage(dicom)
        spacing = tio_image.spacing
        transform = tio.Resample(1)
        res_image = transform(tio_image)
        res_data = torch.movedim(res_image.data, (0,1,2,3), (0,2,1,3)).squeeze()
        with open(f"cache/spc_{patient_id}.pkl", "wb") as f:
            pickle.dump(spacing, f)
        with open(f"cache/dt_{patient_id}.pt", "wb") as file:
            torch.save(res_data, file)     
    else:
        with open(f"cache/dt_{patient_id}.pt", "rb") as file:
            res_data = torch.load(file)
    return res_data


def plot_scan(scan_pt:torch.Tensor, slc:int)->str:
    plt.figure(figsize=(5,5))
    plt.set_cmap('gray')
    plt.imshow(scan_pt[:, :, slc])
    plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", transparent=True)
    buf.seek(0)

    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_str


def process_nodules(nodules: List, PATIENT_ID: str)->Tuple[List, List]:
    with open(f"cache/spc_{PATIENT_ID}.pkl", "rb") as f:
        spacing = pickle.load(f)
    with open(f"cache/dt_{PATIENT_ID}.pt", "rb") as f:
        res_data = torch.load(f)
    NOD_icons_lst = []
    NOD_crops_ref = []
    n=1
    for nodule in nodules:
        #quality control:
        num_annotations=len(nodule)
        if num_annotations > 4:
            continue
        median_malig = np.median([ann.malignancy for ann in nodule])
        if median_malig == 3:
            continue
        if (num_annotations > 2 and num_annotations <= 4):
            cmask, cbbox, _ = consensus(nodule, clevel=0.5)
            res_cbbox = [(round(cbbox[i].start*spacing[i]),
                        round(cbbox[i].stop*spacing[i])) for i in range(3)]
            res_cmask = ndimage.zoom(cmask.astype(int), spacing)
            res_cbbox0 = [round((res_cbbox[i][0]+res_cbbox[i][1])/2) for i in range(3)]
            g = np.zeros(res_data.shape)
            g[res_cbbox[0][0]:res_cbbox[0][0]+res_cmask.shape[0],
            res_cbbox[1][0]:res_cbbox[1][0]+res_cmask.shape[1],
            res_cbbox[2][0]:res_cbbox[2][0]+res_cmask.shape[2],] = res_cmask
            # Nodule surrounding volume of dimmensions (32,32,32)==(2k,2k,2k) is extracted.
            k = int(32/2)
            slices = (
                slice(res_cbbox0[0]-k, res_cbbox0[0]+k),
                slice(res_cbbox0[1]-k, res_cbbox0[1]+k),
                slice(res_cbbox0[2]-k, res_cbbox0[2]+k)
                )
            crop = res_data[slices]

            # it will be used as icon
            central_slc_str = plot_nodule(crop[:, :, 16].unsqueeze(-1))
            NOD_icons_lst.append(central_slc_str)

            with open(f"cache/crops/{PATIENT_ID}_{n}.pt", "wb") as file:
                torch.save(crop, file)
                NOD_crops_ref.append(f"{PATIENT_ID}_{n}")
            n += 1
    return (NOD_icons_lst, NOD_crops_ref)
