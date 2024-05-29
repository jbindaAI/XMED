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
import pandas as pd


app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Loading ground truth data, to compare with predictions:
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


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        name="home.html", request=request, context={"PATIENT_IDs":PATIENT_IDs})



# @app.get("/", response_class=HTMLResponse)
# def home(request: Request):
#     # Loading exemplary original image:
#     original_img = load_img(crop_path="data/crops/0567.pt", crop_view="axial", slice_=15, return_both=False, device="cpu")
#     img_str = plot_nodule(original_img)

#     return templates.TemplateResponse(
#         name="home.html", request=request, context={"NODULE": "0567", "SLC":15, "orig_plot": img_str})


@app.get("/visualize_nodule/{NODULE}", response_class=HTMLResponse)
def visualize_nodule(request: Request, NODULE: str, SLC: int=Query(15, gt=-1, le=31)):
    # loading and visualizing nodule
    original_img = load_img(crop_path=f"data/crops/{NODULE}.pt", crop_view="axial", slice_=SLC, return_both=False, device="cpu")
    img_str = plot_nodule(original_img)

    return templates.TemplateResponse(
        name="home.html", request=request, context={"NODULE": NODULE, "SLC":SLC,"orig_plot": img_str})


@app.get("/predict/{NODULE}/{SLICE}", response_class=HTMLResponse)
def predict(request: Request, NODULE: str, SLICE: int, TASK: str=Query(...)):
    original_img, attention_map, CDAM_maps, model_output = XMED.model_pipeline(NODULE=NODULE, SLICE=SLICE, TASK=TASK)
    img_str = plot_nodule(original_img)
    res_str = plot_results(original=original_img, maps=[attention_map, CDAM_maps])

    # Taking ground truth label:
    nodule_anns = ann_df.loc[ann_df["path"]==NODULE+".pt"]
    if TASK == "Classification":
        GROUND_TRUTH = int(nodule_anns["target"].iloc[0])
        PREDS = round(model_output, 2)
    else:
        features = ['subtlety', 'internalStructure', 'calcification', 'sphericity',
                    'margin', 'lobulation', 'spiculation', 'texture', 'diameter']
        GROUND_TRUTH = nodule_anns[features]
        PREDS = model_output


    return templates.TemplateResponse(
        name="home.html", request=request, context={"NODULE": NODULE, 
                                                    "SLC": SLICE, 
                                                    "TASK": TASK,
                                                    "GROUND_TRUTH": GROUND_TRUTH,
                                                    "PREDS": PREDS,  
                                                    "orig_plot": img_str, 
                                                    "res_plot": res_str})