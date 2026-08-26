import json
import subprocess
from pathlib import Path
from typing import Any

from sqlalchemy.exc import IntegrityError

from fastapi import Cookie, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

from sqlmodel import SQLModel, Session, create_engine, select

import bcrypt
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from models import Drawing, User, get_utctime

SECRET_KEY = "ASFQEUBFOEUQB)!#H) #) UR)(*#!U&R) &)*#!&UR) &$#!)_( &$#)"
serializer = URLSafeTimedSerializer(SECRET_KEY)
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]
SOLVER_BINARY = REPO_ROOT / "build/nonconvex-release/packages/nonconvex-tpp/cpp/tpp"
SOLVER_BUILD_CACHE = REPO_ROOT / "build/nonconvex-release/CMakeCache.txt"


def create_token(username: str) -> str:
    return serializer.dumps(username)


def decode_token(token: str) -> str | None:
    try:
        return serializer.loads(token, max_age=7 * 24 * 3600)  # 7 days
    except (BadSignature, SignatureExpired):
        return None


@asynccontextmanager
async def initFunction(app: FastAPI):
    create_db_and_tables()
    yield


STATIC_PATH = "/static"

app = FastAPI(lifespan=initFunction)
app.mount(STATIC_PATH, StaticFiles(directory=APP_DIR / "static"), name="static")

templates = Jinja2Templates(directory=APP_DIR / "templates")

arquivo_sqlite = "tpp.db"
url_sqlite = f"sqlite:///{arquivo_sqlite}"
engine = create_engine(url_sqlite)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def validate_point(value: Any, name: str) -> tuple[float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(coord, (int, float)) for coord in value)
    ):
        raise HTTPException(status_code=400, detail=f"{name} must be a pair of numbers")

    return float(value[0]), float(value[1])


def validate_polygons(value: Any) -> list[list[tuple[float, float]]]:
    if not isinstance(value, list):
        raise HTTPException(status_code=400, detail="polygons must be a list")

    polygons: list[list[tuple[float, float]]] = []

    for polygon_index, raw_polygon in enumerate(value):
        if not isinstance(raw_polygon, list) or len(raw_polygon) < 3:
            raise HTTPException(
                status_code=400,
                detail=f"polygon {polygon_index} must have at least 3 vertices",
            )

        polygons.append([
            validate_point(point, f"polygons[{polygon_index}] vertex")
            for point in raw_polygon
        ])

    return polygons


def ensure_solver_binary() -> None:
    configured_target = None

    if SOLVER_BUILD_CACHE.exists():
        for line in SOLVER_BUILD_CACHE.read_text().splitlines():
            if line.startswith("TARGET:STRING="):
                configured_target = line.split("=", 1)[1]
                break

    if SOLVER_BINARY.exists() and configured_target == "main-visualizer_solve":
        return

    subprocess.run(
        [
            "cmake",
            "--preset",
            "nonconvex-release",
            "-DTARGET=main-visualizer_solve",
            "-DTPP_ENABLE_GUROBI=OFF",
        ],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", "--preset", "nonconvex-release"],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def solver_input(
    start: tuple[float, float],
    target: tuple[float, float],
    polygons: list[list[tuple[float, float]]],
    max_calls: int,
    max_seconds: float,
) -> str:
    lines = [
        f"{start[0]} {start[1]}",
        f"{target[0]} {target[1]}",
        f"{len(polygons)} {max_calls} {max_seconds}",
    ]

    for polygon in polygons:
        lines.append(str(len(polygon)))
        lines.extend(f"{x} {y}" for x, y in polygon)

    return "\n".join(lines) + "\n"


def parse_solver_output(output: str) -> dict[str, Any]:
    lines = output.strip().splitlines()

    if not lines:
        raise HTTPException(status_code=500, detail="solver produced no output")

    header = lines[0].split()

    if header[0] == "ERR":
        raise HTTPException(status_code=422, detail=" ".join(header[1:]))

    if len(header) != 5 or header[0] != "OK":
        raise HTTPException(status_code=500, detail="invalid solver output")

    exact = header[1] == "1"
    calls = int(header[2])
    seconds = float(header[3])
    point_count = int(header[4])

    if len(lines) != point_count + 1:
        raise HTTPException(status_code=500, detail="truncated solver output")

    path = []
    for line in lines[1:]:
        x, y = line.split()
        path.append([float(x), float(y)])

    return {"path": path, "exact": exact, "calls": calls, "seconds": seconds}


@app.get("/", response_class=HTMLResponse)
async def get_main_page(request: Request, session: str | None = Cookie(default=None)):
    username = decode_token(session) if session else None
    return templates.TemplateResponse(
        request, "index.html", {"static": STATIC_PATH, "username": username}
    )


@app.get("/header", response_class=HTMLResponse)
async def get_header(request: Request, session: str | None = Cookie(default=None)):
    username = decode_token(session) if session else None
    return templates.TemplateResponse(
        request, "header.html", {"static": STATIC_PATH, "username": username}
    )


@app.get("/login", response_class=HTMLResponse)
async def get_login(request: Request):
    return templates.TemplateResponse(
        request, "login.html", {"static": STATIC_PATH}
    )


@app.get("/signup", response_class=HTMLResponse)
async def get_signup(request: Request):
    return templates.TemplateResponse(
        request, "signup.html", {"static": STATIC_PATH}
    )


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_user(username: str, password: str):

    password_hash = hash_password(password)

    with Session(engine) as session:

        user = User(username=username, password_hash=password_hash)
        session.add(user)
        session.commit()
        session.refresh(user)

        return user


@app.post("/signup", response_class=HTMLResponse)
async def post_signup(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
):

    username = username.strip()
    password = password.strip()
    confirm_password = confirm_password.strip()

    def make_error(message: str) -> HTMLResponse:
        return HTMLResponse(
            f'<p style="color: red; font-size: 1.5rem; margin: 0;">{message}</p>'
        )

    if password != confirm_password:
        return make_error("Passwords do not match")

    if len(username) == 0:
        return make_error("Username cannot be empty")

    if len(password) < 1:
        return make_error("Password must be at least 1 characters long")

    if not all(c.isalnum() or c in "_-." for c in username):
        return make_error(
            "Username can only contain letters, numbers, underscores, hyphens and dots"
        )

    if not all(c.isprintable() for c in password):
        return make_error("Password cannot contain non-printable characters")

    if username.startswith(".") or username.endswith("."):
        return make_error("Username cannot start or end with a dot")

    MAX_USERNAME_LENGTH = 100

    if len(username) > MAX_USERNAME_LENGTH:
        return make_error(
            f"Username cannot be longer than {MAX_USERNAME_LENGTH} characters"
        )

    try:
        create_user(username, password)
    except IntegrityError:
        return make_error("Username already exists")
    except Exception as e:
        return make_error("An error occurred while creating the user")

    token = create_token(username)
    response = HTMLResponse(
        "", headers={"HX-Trigger": json.dumps({"loginSuccess": {"username": username}})}
    )
    response.set_cookie("session", token, httponly=True, samesite="lax")

    return response


def authenticate_user(username: str, password: str) -> bool:

    with Session(engine) as session:

        user = session.exec(select(User).where(User.username == username)).first()

        if user is None:
            return False

        return verify_password(password, user.password_hash)


@app.post("/login", response_class=HTMLResponse)
async def post_login(
    request: Request, username: str = Form(...), password: str = Form(...)
):

    username = username.strip()
    password = password.strip()

    if not authenticate_user(username, password):
        return HTMLResponse(
            f'<p style="color: red; font-size: 1.5rem; margin: 0;">Invalid username or password</p>'
        )

    token = create_token(username)
    response = HTMLResponse(
        "", headers={"HX-Trigger": json.dumps({"loginSuccess": {"username": username}})}
    )
    response.set_cookie("session", token, httponly=True, samesite="lax")
    return response


@app.post("/logout", response_class=HTMLResponse)
async def post_logout(request: Request):
    response = HTMLResponse("", headers={"HX-Trigger": "logoutSuccess"})
    response.delete_cookie("session")
    return response


@app.get("/canvas", response_class=HTMLResponse)
async def get_canvas(request: Request):
    return templates.TemplateResponse(
        request, "canvas.html", {"static": STATIC_PATH}
    )


@app.post("/api/tpp/solve")
async def solve_tpp(request: Request):
    data = await request.json()
    start = validate_point(data.get("start"), "start")
    target = validate_point(data.get("target"), "target")
    polygons = validate_polygons(data.get("polygons"))
    max_calls = int(data.get("maxCalls", 200000))
    max_seconds = float(data.get("maxSeconds", 3.0))

    if max_calls < 1 or max_calls > 5000000:
        raise HTTPException(status_code=400, detail="maxCalls is out of range")

    if max_seconds <= 0 or max_seconds > 30:
        raise HTTPException(status_code=400, detail="maxSeconds is out of range")

    try:
        ensure_solver_binary()
        completed = subprocess.run(
            [SOLVER_BINARY],
            input=solver_input(start, target, polygons, max_calls, max_seconds),
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=max_seconds + 5.0,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="solver timed out")
    except subprocess.CalledProcessError as error:
        raise HTTPException(status_code=500, detail=error.stderr.strip() or "solver build failed")

    if completed.returncode != 0 and not completed.stdout:
        raise HTTPException(status_code=500, detail=completed.stderr.strip() or "solver failed")

    return JSONResponse(parse_solver_output(completed.stdout))


@app.get("/save_prompt", response_class=HTMLResponse)
async def get_saved_drawings_prompt(request: Request):
    return templates.TemplateResponse(
        request, "save_prompt.html", {"static": STATIC_PATH}
    )


@app.post("/drawings/save")
async def save_drawing(request: Request, session: str | None = Cookie(default=None)):
    username = decode_token(session) if session else None

    if username is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    data_json = await request.body()

    with Session(engine) as db:
        user = db.exec(select(User).where(User.username == username)).first()

        if not user:
            return HTMLResponse("User not found", status_code=404)

        user_id: int = user.id  # type: ignore
        drawing = Drawing(user_id=user_id, data=data_json.decode())
        db.add(drawing)
        db.commit()
        db.refresh(drawing)

    return JSONResponse({"id": drawing.id})


# Update drawing endpoint
@app.put("/drawings/{drawing_id}")
async def update_drawing(
    drawing_id: int, request: Request, session: str | None = Cookie(default=None)
):

    username = decode_token(session) if session else None

    if username is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    data_json = await request.body()

    with Session(engine) as db:

        user = db.exec(select(User).where(User.username == username)).first()

        if not user:
            return HTMLResponse("User not found", status_code=404)

        drawing = db.exec(
            select(Drawing).where(Drawing.id == drawing_id, Drawing.user_id == user.id)
        ).first()

        if not drawing:
            return HTMLResponse("Drawing not found", status_code=404)

        drawing.data = data_json.decode()

        # Update last modified time
        drawing.modified_at = get_utctime()

        db.add(drawing)
        db.commit()

    return HTMLResponse("OK")


@app.get("/drawings", response_class=HTMLResponse)
async def get_drawings(request: Request, session: str | None = Cookie(default=None)):

    username = decode_token(session) if session else None

    if not username:
        return HTMLResponse("Unauthorized", status_code=401)

    with Session(engine) as db:

        user = db.exec(select(User).where(User.username == username)).first()

        if not user:
            return HTMLResponse("User not found", status_code=404)

        drawings = []

        for d in user.drawings:

            try:
                data = json.loads(d.data)
            except json.JSONDecodeError:

                print(f"Error decoding drawing data for drawing id {d.id}, deleting...")

                db.delete(d)
                db.commit()

                continue

            drawings.append(
                {
                    "id": d.id,
                    "drawing_name": data.get("drawingName", "Untitled Drawing"),
                    "created_at": d.created_at.isoformat(),
                    "modified_at": d.modified_at.isoformat(),
                    "dataURL": data["dataURL"],
                }
            )

        drawings.sort(key=lambda d: d["modified_at"], reverse=True)

    return templates.TemplateResponse(
        request,
        "saved_drawings.html",
        {"static": STATIC_PATH, "drawings": drawings},
    )


@app.get("/drawings/{drawing_id}", response_class=HTMLResponse)
async def get_drawing(
    request: Request, drawing_id: int, session: str | None = Cookie(default=None)
):

    username = decode_token(session) if session else None

    if not username:
        return HTMLResponse("Unauthorized", status_code=401)

    with Session(engine) as db:
        user = db.exec(select(User).where(User.username == username)).first()
        if not user:
            return HTMLResponse("User not found", status_code=404)

        drawing = db.exec(
            select(Drawing).where(Drawing.id == drawing_id, Drawing.user_id == user.id)
        ).first()
        if not drawing:
            return HTMLResponse("Drawing not found", status_code=404)

    data = json.loads(drawing.data)
    return JSONResponse(data)


@app.get("/tpp-info", response_class=HTMLResponse)
async def get_info(request: Request):
    return templates.TemplateResponse(
        request, "tpp_info.html", {"static": STATIC_PATH}
    )


@app.delete("/drawings/{drawing_id}")
async def delete_drawing(drawing_id: int, session: str | None = Cookie(default=None)):

    username = decode_token(session) if session else None

    if username is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    with Session(engine) as db:

        user = db.exec(select(User).where(User.username == username)).first()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        drawing = db.exec(
            select(Drawing).where(Drawing.id == drawing_id, Drawing.user_id == user.id)
        ).first()

        if not drawing:
            raise HTTPException(status_code=404, detail="Drawing not found")

        db.delete(drawing)
        db.commit()

    return HTMLResponse("OK")
