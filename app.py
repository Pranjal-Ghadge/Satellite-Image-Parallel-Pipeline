import csv
import json
import os
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
DATA_INPUT_DIR = PROJECT_DIR / "data" / "input"
DATA_OUTPUT_DIR = PROJECT_DIR / "data" / "output"
FRONTEND_DIR = PROJECT_DIR / "frontend"
BENCHMARK_FILE = PROJECT_DIR / "docs" / "benchmark_results.csv"
PIPELINE_EXE = PROJECT_DIR / "build-ninja" / "parallel_pipeline.exe"
FLOOD_REPORT_FILE = DATA_OUTPUT_DIR / "flood_report.json"
PIPELINE_LOCK = threading.Lock()

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
    pre_filename: str = Field(min_length=1)
    post_filename: str = Field(min_length=1)
    pixel_size: Optional[float] = Field(default=None, gt=0)
    threshold: Optional[float] = Field(default=None, ge=0, le=1)
    auto_threshold: bool = False
    ground_truth_filename: Optional[str] = None


async def save_uploaded_image(upload: UploadFile) -> dict:
    if not upload.filename:
        raise HTTPException(status_code=400, detail="Every uploaded image must have a file name.")

    extension = Path(upload.filename).suffix.lower()
    if extension not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        raise HTTPException(status_code=400, detail=f"Unsupported image format: {extension or 'none'}.")

    saved_path = DATA_INPUT_DIR / f"{uuid.uuid4().hex}{extension}"
    try:
        contents = await upload.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded images cannot be empty.")
        saved_path.write_bytes(contents)
        with Image.open(saved_path) as image:
            image.verify()
    except HTTPException:
        saved_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid image: {upload.filename}") from exc
    finally:
        await upload.close()

    return {
        "filename": saved_path.name,
        "url": f"/input/{saved_path.name}",
        "size_bytes": saved_path.stat().st_size,
    }


@app.post("/api/upload")
async def upload_images(
    pre_image: Optional[UploadFile] = File(default=None),
    post_image: Optional[UploadFile] = File(default=None),
    ground_truth: Optional[UploadFile] = File(default=None),
):
    if (pre_image is None) != (post_image is None):
        raise HTTPException(status_code=400, detail="Upload both pre_image and post_image together.")
    if pre_image is None and post_image is None:
        raise HTTPException(status_code=400, detail="Upload a pre/post image pair.")

    saved = {}
    try:
        if pre_image is not None and post_image is not None:
            saved["pre_image"] = await save_uploaded_image(pre_image)
            saved["post_image"] = await save_uploaded_image(post_image)
        if ground_truth is not None:
            saved["ground_truth"] = await save_uploaded_image(ground_truth)
    except Exception:
        for item in saved.values():
            (DATA_INPUT_DIR / item["filename"]).unlink(missing_ok=True)
        raise

    return {"files": saved}


def resolve_uploaded_path(filename: str, field_name: str) -> Path:
    candidate_name = Path(filename).name
    if candidate_name != filename:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name} filename.")
    candidate = (DATA_INPUT_DIR / candidate_name).resolve()
    if DATA_INPUT_DIR.resolve() not in candidate.parents or not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Uploaded {field_name} image was not found.")
    return candidate


def read_benchmark_csv() -> list[dict]:
    if not BENCHMARK_FILE.exists():
        return []

    results = []
    try:
        with BENCHMARK_FILE.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames and "runtime_ms" in reader.fieldnames:
                for row in reader:
                    try:
                        results.append({
                            "mode": row["mode"],
                            "resolution": row["resolution"],
                            "execution_time_ms": float(row["runtime_ms"]),
                            "throughput_mpps": float(row["throughput_mps"]),
                            "threads": int(row["threads"] or 0),
                            "speedup": float(row["speedup"]),
                            "parallel_efficiency_percent": float(
                                row["parallel_efficiency_percent"]),
                            "verification": row["verification"],
                            "mismatched_flood_pixels": int(
                                row["mismatched_flood_pixels"] or 0),
                            "mismatched_semantic_pixels": int(
                                row["mismatched_semantic_pixels"] or 0),
                        })
                    except (KeyError, TypeError, ValueError):
                        continue
            else:
                file.seek(0)
                for row in csv.reader(file):
                    if len(row) < 4:
                        continue
                    try:
                        results.append({
                            "mode": row[0],
                            "resolution": row[1],
                            "execution_time_ms": float(row[2]),
                            "throughput_mpps": float(row[3]),
                            "threads": 1 if "Sequential" in row[0] else 0,
                            "speedup": None,
                            "parallel_efficiency_percent": None,
                            "verification": "N/A",
                            "mismatched_flood_pixels": 0,
                            "mismatched_semantic_pixels": 0,
                        })
                    except ValueError:
                        continue
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Unable to read benchmark results: {exc}") from exc
    return results


@app.post("/api/run-pipeline")
def run_pipeline(request: PipelineRequest):
    if not PIPELINE_EXE.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Pipeline executable not found: {PIPELINE_EXE}",
        )

    if request.auto_threshold and request.threshold is not None:
        raise HTTPException(
            status_code=400,
            detail="Provide either threshold or auto_threshold, not both.",
        )

    pre_path = resolve_uploaded_path(request.pre_filename, "pre-flood")
    post_path = resolve_uploaded_path(request.post_filename, "post-flood")
    ground_truth_path = None
    if request.ground_truth_filename:
        ground_truth_path = resolve_uploaded_path(
            request.ground_truth_filename, "ground-truth")

    command = [os.path.abspath(PIPELINE_EXE), "--pre", str(pre_path), "--post", str(post_path)]
    if request.pixel_size is not None:
        command.extend(["--pixel-size", str(request.pixel_size)])
    if request.auto_threshold:
        command.append("--auto-threshold")
    elif request.threshold is not None:
        command.extend(["--threshold", str(request.threshold)])
    if ground_truth_path is not None:
        command.extend(["--ground-truth", str(ground_truth_path)])
    command.append("--benchmark")

    try:
        with PIPELINE_LOCK:
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_DIR),
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip() or "Pipeline execution failed."
                raise HTTPException(status_code=500, detail=detail)
            try:
                with FLOOD_REPORT_FILE.open("r", encoding="utf-8") as report_file:
                    flood_report = json.load(report_file)
            except (OSError, json.JSONDecodeError) as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Unable to read generated flood report: {exc}",
                ) from exc
            benchmark_csv_rows = read_benchmark_csv()
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="Pipeline execution timed out.") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Unable to start pipeline: {exc}") from exc

    output_files = {
        "pre_imagery": pre_path.name,
        "post_imagery": post_path.name,
        "permanent_water_mask": "pre_water_mask.png",
        "post_water_mask": "post_water_mask.png",
        "flood_mask": "flood_mask.png",
        "flood_overlay": "flood_overlay.png",
        "semantic_classification": "semantic_classification.png",
    }
    outputs = {}
    for key, filename in output_files.items():
        output_path = DATA_INPUT_DIR / filename if key.endswith("imagery") else DATA_OUTPUT_DIR / filename
        if not output_path.is_file():
            raise HTTPException(status_code=500, detail=f"Expected pipeline output is missing: {filename}")
        directory = "input" if key.endswith("imagery") else "output"
        outputs[key] = f"/{directory}/{filename}?v={uuid.uuid4().hex}"

    benchmark = flood_report.get("benchmark")
    if not isinstance(benchmark, dict):
        raise HTTPException(status_code=500, detail="Flood report is missing benchmark metrics.")
    return {
        "success": True,
        "outputs": outputs,
        "flood_report": flood_report,
        "benchmark": benchmark,
        "benchmark_csv_rows": benchmark_csv_rows[-6:],
        "stdout": completed.stdout,
    }


@app.get("/api/benchmark-results")
def benchmark_results():
    latest_benchmark = None
    if FLOOD_REPORT_FILE.exists():
        try:
            with FLOOD_REPORT_FILE.open("r", encoding="utf-8") as report_file:
                latest_benchmark = json.load(report_file).get("benchmark")
        except (OSError, json.JSONDecodeError):
            latest_benchmark = None
    return JSONResponse(content={
        "results": read_benchmark_csv(),
        "latest_benchmark": latest_benchmark,
    })


app.mount("/output", StaticFiles(directory=DATA_OUTPUT_DIR), name="output")
app.mount("/input", StaticFiles(directory=DATA_INPUT_DIR), name="input")
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
