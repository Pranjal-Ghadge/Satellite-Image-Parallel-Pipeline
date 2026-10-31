import csv
import os
import subprocess
import uuid
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image, ImageOps


PROJECT_DIR = Path(__file__).resolve().parent
DATA_INPUT_DIR = PROJECT_DIR / "data" / "input"
DATA_OUTPUT_DIR = PROJECT_DIR / "data" / "output"
FRONTEND_DIR = PROJECT_DIR / "frontend"
BENCHMARK_FILE = PROJECT_DIR / "docs" / "benchmark_results.csv"
PIPELINE_EXE = PROJECT_DIR / "build-ninja" / "parallel_pipeline.exe"
CMAKE_CACHE = PROJECT_DIR / "build-ninja" / "CMakeCache.txt"

DATA_INPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(PROJECT_DIR / "docs").mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Satellite Image Parallel Pipeline API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PipelineRequest(BaseModel):
    input_filename: Optional[str] = None
    resolution: str = Field(default="1024x1024", pattern=r"^(512x512|1024x1024|4096x4096)$")
    threads: int = Field(default=8, ge=1, le=64)
    enable_cuda: bool = False
    gaussian_kernel: int = Field(default=5, ge=3, le=7)
    omp_schedule: str = Field(default="dynamic", pattern=r"^(static|dynamic)$")


def cuda_build_is_enabled() -> bool:
    if not CMAKE_CACHE.exists():
        return False

    enabled_values = set()
    with CMAKE_CACHE.open("r", encoding="utf-8", errors="replace") as cache:
        for line in cache:
            line = line.strip()
            if line.startswith("USE_CUDA:") or line.startswith("ENABLE_CUDA:"):
                value = line.rsplit("=", 1)[-1].strip().upper()
                enabled_values.add(value == "TRUE" or value == "ON" or value == "1" or value == "YES")

    return bool(enabled_values) and any(enabled_values)


def calculate_speedups_from_stdout(stdout: str) -> dict:
    """Parse runtime blocks emitted by utils.cpp and calculate Sp values."""
    runtimes = {}
    pattern = re.compile(
        r"Execution Mode:\s*(.*?)\s*Resolution:\s*(.*?)\s*Runtime:\s*([\d.]+)\s*ms",
        re.DOTALL,
    )
    for mode, _resolution, runtime_text in pattern.findall(stdout):
        mode_name = mode.strip()
        try:
            runtimes[mode_name] = float(runtime_text)
        except ValueError:
            continue

    baseline = next(
        (
            runtime
            for mode, runtime in runtimes.items()
            if "Sequential" in mode
        ),
        None,
    )
    speedups = {"sequential": 1.0 if baseline else 0.0}
    if baseline and baseline > 0:
        for mode, runtime in runtimes.items():
            if runtime > 0:
                speedups[mode.lower()] = round(baseline / runtime, 2)
    return {
        "runtimes_ms": runtimes,
        "baseline_time_ms": baseline,
        "speedups": speedups,
    }


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="A file name is required.")

    extension = Path(file.filename).suffix.lower()
    if extension not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        raise HTTPException(status_code=400, detail="Unsupported image format.")

    try:
        contents = await file.read()
        upload_id = uuid.uuid4().hex
        saved_path = DATA_INPUT_DIR / f"{upload_id}{extension}"
        saved_path.write_bytes(contents)

        with Image.open(saved_path) as image:
            image.verify()
    except Exception as exc:
        if "saved_path" in locals() and saved_path.exists():
            saved_path.unlink()
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc

    return {
        "filename": saved_path.name,
        "url": f"/input/{saved_path.name}",
        "size_bytes": saved_path.stat().st_size,
    }


@app.post("/api/run-pipeline")
def run_pipeline(request: PipelineRequest):
    if not PIPELINE_EXE.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Pipeline executable not found: {PIPELINE_EXE}",
        )

    cuda_enabled_in_build = cuda_build_is_enabled()
    cuda_requested_but_unavailable = request.enable_cuda and not cuda_enabled_in_build

    if not request.input_filename:
        raise HTTPException(status_code=400, detail="input_filename is required.")

    input_path = (DATA_INPUT_DIR / Path(request.input_filename).name).resolve()
    if DATA_INPUT_DIR.resolve() not in input_path.parents:
        raise HTTPException(status_code=400, detail="Invalid input filename.")
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Uploaded input image was not found.")

    width_height = request.resolution.lower().split("x")
    if len(width_height) != 2:
        raise HTTPException(status_code=400, detail="Invalid resolution.")

    prepared_path = DATA_INPUT_DIR / f"prepared_{input_path.stem}_{width_height[0]}x{width_height[1]}{input_path.suffix.lower()}"
    try:
        with Image.open(input_path) as image:
            grayscale = ImageOps.grayscale(image)
            resized = ImageOps.fit(
                grayscale,
                (int(width_height[0]), int(width_height[1])),
                method=Image.Resampling.LANCZOS,
            )
            if prepared_path.suffix.lower() in {".jpg", ".jpeg"}:
                resized.save(prepared_path, format="JPEG")
            else:
                resized.save(prepared_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Unable to prepare the uploaded image.") from exc

    for output_name in ("seq_edge.jpg", "omp_edge.jpg", "cuda_edge.jpg"):
        output_path = DATA_OUTPUT_DIR / output_name
        if output_path.exists():
            output_path.unlink()

    command = [
        os.path.abspath(PIPELINE_EXE),
        str(prepared_path),
        "--threads",
        str(request.threads),
        "--resolution",
        request.resolution,
    ]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(request.threads)
    environment["PIPELINE_GAUSSIAN_KERNEL"] = str(request.gaussian_kernel)
    environment["PIPELINE_OMP_SCHEDULE"] = request.omp_schedule

    try:
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_DIR),
            env=environment,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="Pipeline execution timed out.") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Unable to start pipeline: {exc}") from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "Pipeline execution failed."
        raise HTTPException(status_code=500, detail=detail)

    outputs = {}
    for key, filename in (
        ("sequential", "seq_edge.jpg"),
        ("openmp", "omp_edge.jpg"),
        ("cuda", "cuda_edge.jpg"),
    ):
        output_path = DATA_OUTPUT_DIR / filename
        if output_path.exists():
            outputs[key] = f"/output/{filename}?v={uuid.uuid4().hex}"

    cuda_image_url = None
    cuda_output = DATA_OUTPUT_DIR / "cuda_edge.jpg"
    if request.enable_cuda and cuda_output.exists():
        cuda_image_url = "/output/cuda_edge.jpg"

    performance = calculate_speedups_from_stdout(completed.stdout)
    normalized_speedups = performance["speedups"]

    return {
        "success": True,
        "input": f"/input/{input_path.name}",
        "prepared_input": f"/input/{prepared_path.name}",
        "resolution": request.resolution,
        "threads": request.threads,
        "cuda_requested": request.enable_cuda,
        "cuda_enabled_in_build": cuda_enabled_in_build,
        "cuda_status": (
            "available" if cuda_image_url
            else "unavailable_build" if cuda_requested_but_unavailable
            else "disabled"
        ),
        "cuda_output_available": "cuda" in outputs,
        "cuda_image_url": cuda_image_url,
        "outputs": outputs,
        "stdout": completed.stdout,
        "baseline_time_ms": performance["baseline_time_ms"],
        "runtimes_ms": performance["runtimes_ms"],
        "speedups": normalized_speedups,
        "speedup": {
            "sequential": normalized_speedups.get("sequential", 1.0),
            "openmp": next(
                (
                    value
                    for mode, value in normalized_speedups.items()
                    if "openmp" in mode
                ),
                1.0,
            ),
            "cuda": normalized_speedups.get(
                next((mode for mode in normalized_speedups if "cuda" in mode), ""),
                1.0 if cuda_image_url else None,
            ),
        },
    }


@app.get("/api/benchmark-results")
def benchmark_results():
    if not BENCHMARK_FILE.exists():
        return JSONResponse(content={"results": []})

    results = []
    baseline_by_resolution = {}
    try:
        with BENCHMARK_FILE.open("r", newline="", encoding="utf-8") as file:
            for row in csv.reader(file):
                if len(row) < 4:
                    continue
                mode, resolution, execution_time, throughput = row[:4]
                try:
                    execution_ms = float(execution_time)
                    throughput_mpps = float(throughput)
                except ValueError:
                    continue
                if not mode.strip():
                    continue
                resolution_key = resolution.strip()
                if "Sequential" in mode:
                    baseline_by_resolution[resolution_key] = execution_ms
                baseline = baseline_by_resolution.get(resolution_key)
                speedup = (
                    round(baseline / execution_ms, 2)
                    if baseline and baseline > 0 and execution_ms > 0
                    else 1.0 if "Sequential" in mode else None
                )
                results.append(
                    {
                        "mode": mode.strip(),
                        "resolution": resolution.strip(),
                        "execution_time_ms": execution_ms,
                        "throughput_mpps": throughput_mpps,
                        "speedup": speedup,
                    }
                )
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Unable to read benchmark results: {exc}") from exc

    return {"results": results}


app.mount("/output", StaticFiles(directory=DATA_OUTPUT_DIR), name="output")
app.mount("/input", StaticFiles(directory=DATA_INPUT_DIR), name="input")
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
