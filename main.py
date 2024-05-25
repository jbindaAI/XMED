from models import XMED
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from loading_data_utils import load_img
import matplotlib.pyplot as plt
import io
import base64
import torch


app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


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
    # Loading exemplary original image:
    original_img = load_img(crop_path="data/crops/0567.pt", crop_view="axial", slice_=15, return_both=False, device="cpu")
    img_str = plot_nodule(original_img)

    return templates.TemplateResponse(
        name="home.html", request=request, context={"NODULE": "0567", "SLC":15, "orig_plot": img_str})


@app.get("/visualize_nodule/{NODULE}", response_class=HTMLResponse)
def visualize_nodule(request: Request, NODULE: str, SLC: int=Query(15, gt=-1, le=32)):
    # loading and visualizing nodule
    original_img = load_img(crop_path=f"data/crops/{NODULE}.pt", crop_view="axial", slice_=SLC, return_both=False, device="cpu")
    img_str = plot_nodule(original_img)

    return templates.TemplateResponse(
        name="home.html", request=request, context={"NODULE": NODULE, "SLC":SLC,"orig_plot": img_str})


@app.get("/predict/{NODULE}", response_class=HTMLResponse)
def predict(request: Request, NODULE: str):
    original_img, attention_map, CDAM_maps = XMED.model_pipeline(NODULE=NODULE, SLICE=16, TASK="Regression")
    print("INFERENCE SUCCESFULL!")
    img_str = plot_nodule(original_img)
    # There is need for continuation...
    
    return templates.TemplateResponse(
        name="home.html", request=request, context={"NODULE": NODULE, "orig_plot": img_str})