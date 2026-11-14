const state = {
    selectedFiles: { pre_image: null, post_image: null, ground_truth: null },
    uploadedFiles: { pre_image: null, post_image: null, ground_truth: null },
    filesChanged: true,
    zoom: 1,
    offsetX: 0,
    offsetY: 0,
    dragging: false,
    dragX: 0,
    dragY: 0,
};

const charts = {};
const uploadFields = [
    { key: "pre_image", input: "#preImageInput", label: "#preFileName", name: "Pre-Flood Satellite Image" },
    { key: "post_image", input: "#postImageInput", label: "#postFileName", name: "Post-Flood Satellite Image" },
    { key: "ground_truth", input: "#groundTruthInput", label: "#groundTruthFileName", name: "Ground-Truth Mask" },
];

function showToast(message) {
    const toast = document.querySelector("#toast");
    toast.textContent = message;
    toast.classList.add("show");
    window.setTimeout(() => toast.classList.remove("show"), 4000);
}

function getApiError(data, fallback) {
    if (!data?.detail) return fallback;
    return typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
}

function appendTerminal(text, marker = ">") {
    const terminal = document.querySelector("#terminalOutput");
    const stamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    terminal.textContent += `[${stamp}] ${marker} ${text}\n`;
    terminal.scrollTop = terminal.scrollHeight;
}

function setProgress(percent, title) {
    document.querySelector("#executionPanel").hidden = false;
    document.querySelector("#progressBar").style.width = `${percent}%`;
    document.querySelector("#executionPercent").textContent = `${percent}%`;
    document.querySelector("#executionTitle").textContent = title;
}

function setImage(imageSelector, downloadSelector, metaSelector, url, filename) {
    const image = document.querySelector(imageSelector);
    const download = document.querySelector(downloadSelector);
    const meta = document.querySelector(metaSelector);
    const frame = image.closest(".image-frame");
    const placeholder = frame?.querySelector(".placeholder");
    image.onload = () => {
        if (meta) {
            meta.textContent = "";
            const dimensions = document.createElement("span");
            dimensions.textContent = `${image.naturalWidth} × ${image.naturalHeight}`;
            meta.append(dimensions);
        }
    };
    image.onerror = () => {
        image.removeAttribute("src");
        image.style.display = "none";
        if (placeholder) placeholder.style.display = "";
        if (meta) meta.textContent = "";
    };

    if (!url) {
        image.removeAttribute("src");
        image.style.display = "none";
        if (placeholder) placeholder.style.display = "";
        if (download) download.hidden = true;
        if (meta) meta.textContent = "";
        return;
    }

    image.src = url;
    image.style.display = "block";
    if (placeholder) placeholder.style.display = "none";
    if (download) {
        download.hidden = false;
        download.href = url;
        download.download = filename;
    }
}

function updateSelectedFile(field) {
    const input = document.querySelector(field.input);
    const file = input.files[0] || null;
    state.selectedFiles[field.key] = file;
    state.uploadedFiles[field.key] = null;
    state.filesChanged = true;
    document.querySelector(field.label).textContent = file ? file.name : field.name;
}

for (const field of uploadFields) {
    document.querySelector(field.input).addEventListener("change", () => updateSelectedFile(field));
}

async function uploadSelection() {
    if (!state.selectedFiles.pre_image || !state.selectedFiles.post_image) {
        throw new Error("Choose both a pre-flood and post-flood image.");
    }

    const body = new FormData();
    body.append("pre_image", state.selectedFiles.pre_image);
    body.append("post_image", state.selectedFiles.post_image);
    if (state.selectedFiles.ground_truth) {
        body.append("ground_truth", state.selectedFiles.ground_truth);
    }

    const response = await fetch("/api/upload", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(getApiError(data, "Upload failed"));
    state.uploadedFiles = {
        pre_image: data.files.pre_image.filename,
        post_image: data.files.post_image.filename,
        ground_truth: data.files.ground_truth?.filename || null,
    };
    state.filesChanged = false;

    setImage("#preImage", "#downloadPre", "#preMeta", data.files.pre_image.url, "pre_flood_image");
    setImage("#postImage", "#downloadPost", "#postMeta", data.files.post_image.url, "post_flood_image");
    appendTerminal("Pre/post images uploaded and validated.", "[OK]");
    if (data.files.ground_truth) appendTerminal("Ground-truth mask uploaded.", "[OK]");
    return data;
}

document.querySelector("#uploadButton").addEventListener("click", async () => {
    const button = document.querySelector("#uploadButton");
    button.disabled = true;
    try {
        appendTerminal("Uploading the image pair and optional reference mask...");
        const uploaded = await uploadSelection();
        if (uploaded.files.ground_truth) {
            document.querySelector("#groundTruthFileName").textContent = "Reference mask uploaded";
        }
        showToast("Image pair uploaded successfully");
    } catch (error) {
        appendTerminal(error.message, "[ERR]");
        showToast(error.message);
    } finally {
        button.disabled = false;
    }
});

document.querySelector("#autoThreshold").addEventListener("change", event => {
    document.querySelector("#threshold").disabled = event.target.checked;
});

function applySyncedTransform() {
    document.querySelectorAll(".image-frame img").forEach(image => {
        image.style.transform = `translate(${state.offsetX}px, ${state.offsetY}px) scale(${state.zoom})`;
    });
}

document.querySelectorAll(".image-frame").forEach(frame => {
    frame.addEventListener("wheel", event => {
        event.preventDefault();
        state.zoom = Math.min(4, Math.max(1, state.zoom * (event.deltaY > 0 ? 0.9 : 1.1)));
        if (state.zoom === 1) {
            state.offsetX = 0;
            state.offsetY = 0;
        }
        applySyncedTransform();
    }, { passive: false });
    frame.addEventListener("pointerdown", event => {
        if (state.zoom === 1) return;
        state.dragging = true;
        state.dragX = event.clientX - state.offsetX;
        state.dragY = event.clientY - state.offsetY;
        frame.setPointerCapture(event.pointerId);
    });
    frame.addEventListener("pointermove", event => {
        if (!state.dragging) return;
        state.offsetX = event.clientX - state.dragX;
        state.offsetY = event.clientY - state.dragY;
        applySyncedTransform();
    });
    frame.addEventListener("pointerup", () => { state.dragging = false; });
    frame.addEventListener("click", () => {
        const image = frame.querySelector("img");
        if (!image.src) return;
        document.querySelector("#modalImage").src = image.src;
        document.querySelector("#imageModal").classList.add("open");
        document.querySelector("#imageModal").setAttribute("aria-hidden", "false");
    });
});

document.querySelector("#resetZoomButton").addEventListener("click", () => {
    state.zoom = 1;
    state.offsetX = 0;
    state.offsetY = 0;
    applySyncedTransform();
});
document.querySelector("#closeModal").addEventListener("click", () => {
    document.querySelector("#imageModal").classList.remove("open");
    document.querySelector("#imageModal").setAttribute("aria-hidden", "true");
});
document.querySelector("#imageModal").addEventListener("click", event => {
    if (event.target.id === "imageModal") document.querySelector("#closeModal").click();
});

function renderMetrics(report) {
    const areas = report.areas;
    const benchmark = report.benchmark;
    const runs = benchmark.openmp_runs || [];
    document.querySelector("#floodArea").textContent = `${Number(areas.newly_inundated_flood_km2).toFixed(4)} km²`;
    document.querySelector("#floodAreaHa").textContent = `${Number(areas.newly_inundated_flood_hectares).toFixed(2)} hectares`;
    document.querySelector("#waterBreakdown").textContent =
        `${Number(areas.permanent_water_km2).toFixed(3)} / ${Number(areas.newly_inundated_flood_km2).toFixed(3)} km²`;
    document.querySelector("#waterBreakdownDetail").textContent =
        `Permanent / newly flooded · post-flood total ${Number(areas.post_flood_total_water_km2).toFixed(3)} km²`;

    const sequential = Number(benchmark.sequential_runtime_ms);
    const fastest = runs.reduce((best, run) => !best || run.runtime_ms < best.runtime_ms ? run : best, null);
    document.querySelector("#sequentialTime").textContent = `${sequential.toFixed(2)} ms`;
    document.querySelector("#parallelTime").textContent = fastest ? `${Number(fastest.runtime_ms).toFixed(2)} ms` : "—";
    document.querySelector("#parallelSummary").textContent = fastest
        ? `Fastest OpenMP run · ${fastest.threads} threads`
        : "No OpenMP benchmark runs";
    const peak = runs.reduce((best, run) => !best || run.speedup > best.speedup ? run : best, null);
    document.querySelector("#peakSpeedup").textContent = peak ? `${Number(peak.speedup).toFixed(2)}×` : "—";
    document.querySelector("#parallelEfficiency").textContent = peak
        ? `${Number(peak.parallel_efficiency_percent).toFixed(1)}% efficiency · ${peak.threads} threads`
        : "Peak speedup / efficiency";

    const verified = runs.length > 0 && runs.every(run => run.verification_passed);
    const mismatches = runs.reduce((total, run) =>
        total + Number(run.mismatched_pixels || 0) + Number(run.semantic_mismatched_pixels || 0), 0);
    const verification = document.querySelector("#verificationStatus");
    verification.textContent = verified ? "PASSED / 0 mismatched pixels" : `FAILED / ${mismatches} mismatched pixels`;
    verification.classList.toggle("passed", verified);
    verification.classList.toggle("failed", !verified);
    document.querySelector("#verificationDetail").textContent =
        verified ? "Flood and semantic maps match across all OpenMP runs" : "Flood and semantic map equivalence check";

    renderCharts(runs, sequential);
}

function renderCharts(runs, sequential) {
    if (!window.Chart) {
        showToast("Chart.js is unavailable; flood maps and metrics are still ready.");
        return;
    }
    const labels = runs.map(run => `${run.threads} thread${run.threads === 1 ? "" : "s"}`);
    const times = runs.map(run => Number(run.runtime_ms));
    const speedups = runs.map(run => Number(run.speedup));
    const efficiencies = runs.map(run => Number(run.parallel_efficiency_percent));
    Chart.defaults.color = "#94a3b8";
    Chart.defaults.borderColor = "rgba(148,163,184,.12)";
    Chart.defaults.font.family = "Inter, Segoe UI, sans-serif";

    const specifications = [
        ["time", "timeChart", "bar", "Runtime (ms)", times, "#22d3ee"],
        ["speedup", "speedupChart", "line", "Speedup (×)", speedups, "#a78bfa"],
        ["efficiency", "efficiencyChart", "line", "Efficiency (%)", efficiencies, "#34d399"],
    ];
    for (const [key, canvasId, type, label, values, color] of specifications) {
        const canvas = document.querySelector(`#${canvasId}`);
        const options = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true }, x: { grid: { display: false } } },
        };
        const data = {
            labels,
            datasets: [{
                label,
                data: values,
                borderColor: color,
                backgroundColor: type === "bar" ? `${color}cc` : `${color}22`,
                fill: type === "line",
                tension: 0.3,
                pointRadius: 4,
                borderRadius: type === "bar" ? 6 : 0,
            }],
        };
        if (charts[key]) {
            charts[key].data = data;
            charts[key].options = options;
            charts[key].update();
        } else {
            charts[key] = new Chart(canvas, { type, data, options });
        }
    }
    document.querySelector("#timeChart").dataset.baselineMs = String(sequential);
}

function displayOutputs(outputs) {
    setImage("#preImage", "#downloadPre", "#preMeta", outputs.pre_imagery, "pre_flood_image");
    setImage("#postImage", "#downloadPost", "#postMeta", outputs.post_imagery, "post_flood_image");
    setImage("#permanentMask", "#downloadPermanent", "#permanentMeta", outputs.permanent_water_mask, "permanent_water_mask.png");
    setImage("#floodMask", "#downloadFlood", "#floodMeta", outputs.flood_mask, "flood_mask.png");
    setImage("#floodOverlay", "#downloadOverlay", "#overlayMeta", outputs.flood_overlay, "flood_overlay.png");
}

document.querySelector("#runButton").addEventListener("click", async () => {
    const runButton = document.querySelector("#runButton");
    runButton.disabled = true;
    document.querySelector("#systemStatus").textContent = "Processing";
    document.querySelector(".status-pill").classList.add("processing");
    document.querySelector("#terminalOutput").textContent = "";
    try {
        if (state.filesChanged || !state.uploadedFiles.pre_image || !state.uploadedFiles.post_image) {
            setProgress(10, "Uploading satellite image pair");
            await uploadSelection();
        }

        const autoThreshold = document.querySelector("#autoThreshold").checked;
        const pixelText = document.querySelector("#pixelSize").value.trim();
        const thresholdText = document.querySelector("#threshold").value.trim();
        const payload = {
            pre_filename: state.uploadedFiles.pre_image,
            post_filename: state.uploadedFiles.post_image,
            ground_truth_filename: state.uploadedFiles.ground_truth,
            pixel_size: pixelText ? Number(pixelText) : null,
            auto_threshold: autoThreshold,
            threshold: autoThreshold ? null : Number(thresholdText),
        };
        if (pixelText && (!Number.isFinite(payload.pixel_size) || payload.pixel_size <= 0)) {
            throw new Error("Pixel size must be a positive number.");
        }
        if (!autoThreshold && (!Number.isFinite(payload.threshold) || payload.threshold < 0 || payload.threshold > 1)) {
            throw new Error("Threshold must be between 0 and 1.");
        }

        setProgress(25, "Running sequential and OpenMP pipelines");
        appendTerminal("Running complete flood pipeline with 1, 2, 4, 8, and 16 OpenMP threads.", "[RUN]");
        appendTerminal(autoThreshold ? "Using independent Otsu thresholds." : `Using water threshold ${payload.threshold}.`);
        const response = await fetch("/api/run-pipeline", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(getApiError(data, "Flood analysis failed"));

        setProgress(78, "Loading flood maps and reports");
        displayOutputs(data.outputs);
        renderMetrics(data.flood_report);
        appendTerminal("Flood report, benchmark CSV, and output maps loaded.", "[OK]");
        appendTerminal(`Sequential runtime: ${Number(data.benchmark.sequential_runtime_ms).toFixed(2)} ms.`);
        const checked = (data.benchmark.openmp_runs || []).filter(run => run.verification_passed);
        appendTerminal(`Sequential/OpenMP verification: ${checked.length}/${data.benchmark.openmp_runs.length} runs passed.`, checked.length === data.benchmark.openmp_runs.length ? "[OK]" : "[WARN]");
        const accuracy = data.flood_report.accuracy_metrics;
        if (accuracy?.evaluated) {
            appendTerminal(`Ground-truth F1 ${Number(accuracy.f1_score_dice_coefficient).toFixed(4)} · IoU ${Number(accuracy.intersection_over_union_critical_success_index).toFixed(4)}.`);
        }
        setProgress(100, "Flood analysis complete");
        showToast("Flood inundation mapping complete");
    } catch (error) {
        appendTerminal(error.message, "[ERR]");
        showToast(error.message);
        setProgress(100, "Execution failed");
    } finally {
        runButton.disabled = false;
        document.querySelector("#systemStatus").textContent = "System Ready";
        document.querySelector(".status-pill").classList.remove("processing");
    }
});

async function loadPreviousBenchmarks() {
    try {
        const response = await fetch("/api/benchmark-results");
        const data = await response.json();
        const runs = data.latest_benchmark?.openmp_runs || [];
        const sequential = Number(data.latest_benchmark?.sequential_runtime_ms);
        if (runs.length) renderCharts(runs, sequential);
    } catch (_) {
        showToast("Unable to load previous benchmark analytics.");
    }
}

document.querySelectorAll("img").forEach(image => {
    image.addEventListener("error", () => {
        image.removeAttribute("src");
        image.style.display = "none";
        const placeholder = image.closest(".image-frame")?.querySelector(".placeholder");
        if (placeholder) placeholder.style.display = "";
    });
});

loadPreviousBenchmarks();
