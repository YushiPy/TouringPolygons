"""Read-only event preview for phones on the local network."""
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dashboard.dashboard_event import event_context, inline_event_assets

ROOT = Path(__file__).resolve().parent
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))


@app.get("/")
async def index():
	return RedirectResponse("/evento")


@app.get("/evento")
async def event(request: Request):
	return templates.TemplateResponse(request, "event.html", {"standalone": True, **event_context()})


@app.get("/evento/offline", response_class=HTMLResponse)
async def offline(request: Request):
	html = templates.get_template("event.html").render(request=request, offline=True, **event_context())
	return HTMLResponse(inline_event_assets(html), headers={"Content-Disposition": 'attachment; filename="tpp-siicusp34.html"'})
