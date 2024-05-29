from models import XMED
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from loading_data_utils import load_img
from models.att_cdam_utils import get_cmap
import matplotlib.pyplot as plt
import pickle
import io
import base64
import torch
import pylidc as pl
from pylidc.utils import consensus
from scipy import ndimage
import torchio as tio
import os
import numpy as np
from typing import List, Tuple

# Cleaning cache:
for file in os.listdir("cache/"):
    if file != "crops":
        os.remove(f"cache/{file}")
        
for file in os.listdir("cache/crops/"):
    os.remove(f"cache/crops/{file}")


app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Loading annotations for all dataset
with open("data/ALL_annotations_df.pkl", "rb") as file:
    ann_df = pickle.load(file)

# Loading list of available Patients:
with open("data/match_ALL_df.pkl", "rb") as file:
    match_df = pickle.load(file)
    PATIENT_IDs = list(match_df[0])
    PATIENT_IDs = list(dict.fromkeys(PATIENT_IDs))


def plot_results(original, maps):
    """Using matplotlib, plot the original image and the relevance maps"""
    maps2plot=[]
    target_names=[]
    maps2plot.append(maps[0])
    target_names.append("Attention Map")
    for key in maps[1].keys():
        maps2plot.append(maps[1][key])
        target_names.append(key)

    if len(maps2plot) == 2:
        # Binary classification:
        plt.figure(figsize=(12.5, 3.75))
        num_plots = 3
        plt.subplot(1, num_plots, 1)
        plt.imshow(original[:, :, 0])
        plt.set_cmap('gray')
        plt.axis("off")
        for i, m in enumerate(maps2plot):
            plt.subplot(1, num_plots, i + 2)
            plt.imshow(m, cmap=get_cmap(m))
            plt.axis("off")
        plt.subplots_adjust(wspace=0.005, hspace=0)

    elif len(maps2plot) == 9:
        # Biomarker Regression:
        fig, axs = plt.subplots(2, 5, figsize=(12.5, 5))

        # Plotting attention map and CDAM scores:
        k = 0
        for i in range(2):
            for j in range(5):
                if i == 0 and j == 0:
                    axs[0][0].imshow(original, cmap='gray')
                    axs[0][0].set_title("Original Image")
                else:
                    axs[i][j].imshow(maps2plot[k], cmap=get_cmap(maps2plot[k]))
                    axs[i][j].set_title(target_names[k])
                    k += 1
        fig.tight_layout()

    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", transparent=True)
    buf.seek(0)

    res_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return res_str

## Tutaj należałoby trochę poprawić, by lepiej współpracowało z process_nodule
def plot_nodule(original_img:torch.Tensor)->str:
    plt.figure(figsize=(3.5,3.5))
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


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        name="home.html", request=request, context={"PATIENT_IDs":PATIENT_IDs})


@app.get("/visualize_scan/{PATIENT_ID}", response_class=HTMLResponse)
def visualize_scan(request: Request, PATIENT_ID: str, SLC: int=Query(150)):
    scan_pt = load_dicom_into_tensor(patient_id=PATIENT_ID)
    max_depth = scan_pt.shape[-1]
    scan_str = plot_scan(scan_pt=scan_pt, slc=SLC)

    return templates.TemplateResponse(
        name="scan_slicer.html", request=request, context={"PATIENT_IDs":PATIENT_IDs,
                                                    "PATIENT_ID":PATIENT_ID, 
                                                    "SLC": SLC,
                                                    "max_depth": max_depth,
                                                    "scan_plot": scan_str})


@app.get("/extract_nodules/{PATIENT_ID}", response_class=HTMLResponse)
def extract_nodules(request: Request, PATIENT_ID: str):
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == f"LIDC-IDRI-{PATIENT_ID}").first()
    nodules = scan.cluster_annotations()
    NOD_icons_lst, NOD_crops_ref = process_nodules(nodules=nodules, PATIENT_ID=PATIENT_ID)
    NODULES=zip(NOD_icons_lst, NOD_crops_ref)
    # caching NODULES to retrieve later:
    with open("cache/nodules_lst.pkl", "wb") as file:
        pickle.dump(NODULES, file)

    return templates.TemplateResponse(
        name="nodules_list.html", request=request, context={"PATIENT_IDs":PATIENT_IDs,
                                                    "NODULES": NODULES
                                                    })


@app.get("/list_nodules/", response_class=HTMLResponse)
def list_nodules(request: Request):
    with open("cache/nodules_lst.pkl", "rb") as file:
        NODULES = pickle.load(file)
    return templates.TemplateResponse(
        name="nodules_list.html", request=request, context={"PATIENT_IDs":PATIENT_IDs,
                                                    "NODULES": NODULES
                                                    })


@app.get("/visualize_nodule/{NOD_crop}", response_class=HTMLResponse)
def visualize_nodule(request: Request, NOD_crop: str, SLC: int=Query(15, gt=-1, le=31)):
    # when user decide to analyze Nodule, it deletes cached tensors of patients scans other than chosen one.
    cached_files = os.listdir("cache/")
    for file in cached_files:
        if file != "crops" and file != "nodules_lst.pkl":
            os.remove("cache/"+file)
    # loading and visualizing nodule
    original_img = load_img(crop_path=f"cache/crops/{NOD_crop}.pt", crop_view="axial", slice_=SLC, return_both=False, device="cpu")
    img_str = plot_nodule(original_img)

    # Context to return to Nodule lst:
    with open("cache/nodules_lst.pkl", "rb") as file:
        NODULES = pickle.load(file)

    return templates.TemplateResponse(
        name="nodule_slicer.html", request=request, context={"NOD_crop": NOD_crop,
                                                             "SLC":SLC,
                                                             "NODULES": NODULES,
                                                             "orig_plot": img_str})


@app.get("/predict/{NODULE}/{SLICE}", response_class=HTMLResponse)
def predict(request: Request, NODULE: str, SLICE: int, TASK: str=Query(...)):
    original_img, attention_map, CDAM_maps, model_output = XMED.model_pipeline(NODULE=NODULE, SLICE=SLICE, TASK=TASK)
    img_str = plot_nodule(original_img)
    res_str = plot_results(original=original_img, maps=[attention_map, CDAM_maps])

    if TASK == "Classification":
        PREDS = round(model_output, 2)
    else:
        PREDS = model_output


    return templates.TemplateResponse(
        name="nodule_slicer.html", request=request, context={"NOD_crop": NODULE, 
                                                    "SLC": SLICE, 
                                                    "TASK": TASK,
                                                    "PREDS": PREDS,  
                                                    "orig_plot": img_str, 
                                                    "res_plot": res_str})