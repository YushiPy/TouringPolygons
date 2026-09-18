"""Read-only event preview for phones on the local network."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dashboard.dashboard_event import event_context, german_context, inline_event_assets, inline_german_assets

ROOT = Path(__file__).resolve().parent
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))


@app.get("/")
async def index():
    return RedirectResponse("/evento")


@app.get("/evento")
async def event(request: Request):
    return templates.TemplateResponse(request, "german.html", {"standalone": True, **german_context()})


@app.get("/evento/alemao")
async def german_event(request: Request):
    return templates.TemplateResponse(request, "german.html", {"standalone": True, **german_context()})


@app.get("/evento/offline", response_class=HTMLResponse)
async def offline(request: Request):
    html = templates.get_template("german.html").render(request=request, offline=True, standalone=True, **german_context())
    return HTMLResponse(
        inline_german_assets(html), headers={"Content-Disposition": 'attachment; filename="tpp-corpus-alemao-558.html"'}
    )


@app.get("/evento/siicusp")
async def siicusp_event(request: Request):
    return templates.TemplateResponse(request, "event.html", {"archive": True, "standalone": True, **event_context()})


@app.get("/evento/siicusp/offline", response_class=HTMLResponse)
async def siicusp_offline(request: Request):
    html = templates.get_template("event.html").render(request=request, offline=True, archive=True, **event_context())
    return HTMLResponse(
        inline_event_assets(html), headers={"Content-Disposition": 'attachment; filename="tpp-siicusp34.html"'}
    )
