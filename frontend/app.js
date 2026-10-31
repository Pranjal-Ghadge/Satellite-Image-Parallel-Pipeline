const state = {
    uploadedFilename: null,
    zoom: 1,
    offsetX: 0,
    offsetY: 0,
    dragging: false,
    dragX: 0,
    dragY: 0,
};

document.querySelectorAll("img").forEach(image => {
    image.addEventListener("error", () => {
        image.removeAttribute("src");
        image.style.display = "none";
        const frame = image.closest(".image-frame");
        if (frame) {
            const placeholder = frame.querySelector(".placeholder");
            if (placeholder) placeholder.style.display = "";
        }
    });
});

const charts = {};
const elements = {
    uploadZone: document.querySelector("#uploadZone"),
    imageInput: document.querySelector("#imageInput"),
    runButton: document.querySelector("#runButton"),
    resolution: document.querySelector("#resolution"),
    threads: document.querySelector("#threads"),
    cuda: document.querySelector("#cudaToggle"),
};

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

function setSpeedupBadge(id, speedup, unavailableText = "Awaiting run") {
    const badge = document.querySelector(id);
    if (!badge) return;
    if (speedup === null || speedup === undefined) {
        badge.textContent = unavailableText;
        return;
    }
    badge.textContent = `${Number(speedup).toFixed(2)}x Speedup`;
}

async function updateMetadata(elementId, metaId, url) {
    const image = document.querySelector(elementId);
    const meta = document.querySelector(metaId);
    if (!url) {
        meta.innerHTML = "";
        return;
    }

    await new Promise(resolve => {
        if (image.complete) resolve();
        else image.onload = resolve;
    });

    let size = "Size —";
    try {
        const response = await fetch(url, { method: "HEAD" });
        const bytes = Number(response.headers.get("content-length"));
        if (Number.isFinite(bytes) && bytes > 0) size = bytes < 1048576 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1048576).toFixed(1)} MB`;
    } catch (_) {}

    meta.innerHTML = `<span>${image.naturalWidth || 0} × ${image.naturalHeight || 0}</span><span>${size}</span><span>Grayscale</span>`;
}

function setImage(elementId, downloadId, metaId, url, filename) {
    const image = document.querySelector(elementId);
    const download = document.querySelector(downloadId);
    const card = image.closest(".image-card") || image.parentElement.parentElement;
    const placeholder = card?.querySelector(".placeholder");

    if (url) {
        image.src = url;
    } else {
        image.removeAttribute("src");
    }
    image.style.display = url ? "block" : "none";
    if (placeholder) placeholder.style.display = url ? "none" : "";
    if (url) {
        download.hidden = false;
        download.href = url;
        download.download = filename;
        updateMetadata(elementId, metaId, url);
    } else {
        download.hidden = true;
    }
}

async function uploadFile(file) {
    if (!file) return;
    const body = new FormData();
    body.append("file", file);
    appendTerminal("Validating and uploading image...");
    try {
        const response = await fetch("/api/upload", { method: "POST", body });
        const data = await response.json();
        if (!response.ok) throw new Error(getApiError(data, "Upload failed"));
        state.uploadedFilename = data.filename;
        setImage("#inputImage", "#downloadInput", "#inputMeta", `${data.url}?t=${Date.now()}`, data.filename);
        appendTerminal(`Upload complete: ${data.filename} (${(data.size_bytes / 1024).toFixed(1)} KB)`, "[OK]");
        showToast("Image uploaded successfully");
    } catch (error) {
        appendTerminal(error.message, "[ERR]");
        showToast(error.message);
    }
}

elements.uploadZone.addEventListener("click", () => elements.imageInput.click());
elements.imageInput.addEventListener("change", () => uploadFile(elements.imageInput.files[0]));
["dragenter", "dragover"].forEach(type => elements.uploadZone.addEventListener(type, event => {
    event.preventDefault();
    elements.uploadZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach(type => elements.uploadZone.addEventListener(type, event => {
    event.preventDefault();
    elements.uploadZone.classList.remove("dragging");
}));
elements.uploadZone.addEventListener("drop", event => uploadFile(event.dataTransfer.files[0]));

elements.cuda.addEventListener("change", () => {
    const cudaPlaceholder = document.querySelector("#cudaPlaceholder");
    if (!elements.cuda.checked) {
        document.querySelector("#cuda-img").removeAttribute("src");
        document.querySelector("#cuda-img").style.display = "none";
        cudaPlaceholder.style.display = "";
        cudaPlaceholder.textContent = "CUDA Disabled";
        document.querySelector("#cudaSpeedup").textContent = "CUDA Disabled";
        document.querySelector("#downloadCuda").hidden = true;
        document.querySelector("#cudaMeta").innerHTML = "";
    } else {
        cudaPlaceholder.textContent = "Awaiting CUDA Run";
    }
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
    frame.addEventListener("click", event => {
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

function updateSplit() {
    const viewer = document.querySelector("#splitViewer");
    const after = document.querySelector("#splitAfter");
    const selected = document.querySelector("#splitProcessed").value;
    const selectedImage = document.querySelector(`#${selected}`);
    const splitBefore = document.querySelector("#splitBefore");
    const beforeSrc = document.querySelector("#inputImage").src || "";
    let hasComparisonImage = false;
    if (document.querySelector("#inputImage").hasAttribute("src") && beforeSrc) {
        splitBefore.src = beforeSrc;
        splitBefore.style.display = "block";
        hasComparisonImage = true;
    } else {
        splitBefore.removeAttribute("src");
        splitBefore.style.display = "none";
    }
    if (selectedImage?.hasAttribute("src") && selectedImage.src) {
        after.src = selectedImage.src;
        after.style.display = "block";
    } else {
        after.removeAttribute("src");
        after.style.display = "none";
    }
    viewer.classList.toggle("has-image", hasComparisonImage);
    after.style.width = `${viewer.clientWidth}px`;
}

function setSplitPosition(clientX) {
    const viewer = document.querySelector("#splitViewer");
    const rect = viewer.getBoundingClientRect();
    const position = Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100));
    document.querySelector("#splitAfterWrap").style.right = `${100 - position}%`;
    document.querySelector("#splitDivider").style.left = `${position}%`;
}

document.querySelector("#splitViewer").addEventListener("pointerdown", event => {
    event.currentTarget.setPointerCapture(event.pointerId);
    setSplitPosition(event.clientX);
});
document.querySelector("#splitViewer").addEventListener("pointermove", event => {
    if (event.buttons === 1) setSplitPosition(event.clientX);
});
document.querySelector("#splitProcessed").addEventListener("change", updateSplit);

const drawer = document.querySelector("#advancedDrawer");
const backdrop = document.querySelector("#drawerBackdrop");
function toggleDrawer(open) {
    drawer.classList.toggle("open", open);
    backdrop.classList.toggle("open", open);
    drawer.setAttribute("aria-hidden", String(!open));
}
document.querySelector("#advancedButton").addEventListener("click", () => toggleDrawer(true));
document.querySelector("#closeAdvanced").addEventListener("click", () => toggleDrawer(false));
backdrop.addEventListener("click", () => toggleDrawer(false));

elements.runButton.addEventListener("click", async () => {
    if (!state.uploadedFilename) {
        showToast("Upload an image before running the pipeline.");
        return;
    }

    elements.runButton.disabled = true;
    elements.runButton.textContent = "Processing...";
    document.querySelector("#systemStatus").textContent = "Processing";
    document.querySelector(".status-pill").classList.add("processing");
    document.querySelector("#terminalOutput").textContent = "";
    setProgress(10, "Preparing pipeline");
    appendTerminal("[1/3] Upload prepared image", "[RUN]");

    await new Promise(resolve => setTimeout(resolve, 180));
    setProgress(38, "Running CPU kernels");
    appendTerminal("[2/3] Running sequential/OpenMP kernels");
    if (elements.cuda.checked) appendTerminal("CUDA pass enabled; waiting for GPU executable output");
    appendTerminal(`Kernel ${document.querySelector("#kernelSize").value}×${document.querySelector("#kernelSize").value} filter; scheduling=${document.querySelector("#ompSchedule").value}`);

    try {
        const response = await fetch("/api/run-pipeline", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                input_filename: state.uploadedFilename,
                resolution: elements.resolution.value,
                threads: Number(elements.threads.value),
                enable_cuda: elements.cuda.checked,
                gaussian_kernel: Number(document.querySelector("#kernelSize").value),
                omp_schedule: document.querySelector("#ompSchedule").value,
            }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(getApiError(data, "Pipeline failed"));

        setProgress(78, "Parsing benchmark results");
        appendTerminal("[3/3] Loading output maps and parsing benchmark CSV");
        setImage("#inputImage", "#downloadInput", "#inputMeta", data.prepared_input, "input_prepared.jpg");
        setImage("#sequentialImage", "#downloadSequential", "#sequentialMeta", data.outputs.sequential, "sequential_edge.jpg");
        setImage("#openmpImage", "#downloadOpenmp", "#openmpMeta", data.outputs.openmp, "openmp_edge.jpg");
        document.querySelector("#sequentialSpeedup").textContent = "1.00x (Baseline)";
        setSpeedupBadge("#openmpSpeedup", data.speedup?.openmp);

        const cudaImage = document.querySelector("#cuda-img");
        const cudaPlaceholder = document.querySelector("#cudaPlaceholder");
        if (data.cuda_image_url) {
            cudaPlaceholder.style.display = "none";
            cudaImage.style.display = "block";
            cudaImage.src = `${data.cuda_image_url}?t=${Date.now()}`;
            document.querySelector("#downloadCuda").hidden = false;
            document.querySelector("#downloadCuda").href = data.cuda_image_url;
            updateMetadata("#cuda-img", "#cudaMeta", data.cuda_image_url);
            setSpeedupBadge("#cudaSpeedup", data.speedup?.cuda, "CUDA Unavailable");
            appendTerminal("CUDA edge map generated", "[OK]");
        } else if (data.cuda_requested) {
            cudaImage.removeAttribute("src");
            cudaImage.style.display = "none";
            cudaPlaceholder.style.display = "";
            cudaPlaceholder.textContent = data.cuda_status === "unavailable_build" ? "CUDA Not Built" : "CUDA Unavailable";
            document.querySelector("#cudaMeta").innerHTML = "";
            document.querySelector("#cudaSpeedup").textContent = "CUDA Unavailable";
            appendTerminal("CUDA requested, but the current executable is CPU-only", "[WARN]");
        } else {
            cudaImage.removeAttribute("src");
            cudaImage.style.display = "none";
            cudaPlaceholder.style.display = "";
            cudaPlaceholder.textContent = "CUDA Disabled";
            document.querySelector("#cudaSpeedup").textContent = "CUDA Disabled";
            appendTerminal("CUDA pass skipped", "[INFO]");
        }

        updateSplit();
        await loadBenchmarks();
        setProgress(100, "Benchmark complete");
        appendTerminal("Pipeline and benchmark completed successfully", "[OK]");
        showToast(data.cuda_image_url ? "Pipeline and CUDA execution completed" : "Pipeline completed");
    } catch (error) {
        appendTerminal(error.message, "[ERR]");
        showToast(error.message);
        setProgress(100, "Execution failed");
    } finally {
        elements.runButton.disabled = false;
        elements.runButton.textContent = "Run Benchmark";
        document.querySelector("#systemStatus").textContent = "System Ready";
        document.querySelector(".status-pill").classList.remove("processing");
    }
});

function chartDefaults() {
    Chart.defaults.color = "#94a3b8";
    Chart.defaults.borderColor = "rgba(148,163,184,.12)";
    Chart.defaults.font.family = "Inter, Segoe UI, sans-serif";
}

const speedupDataLabelPlugin = {
    id: "speedupDataLabels",
    afterDatasetsDraw(chart) {
        const { ctx } = chart;
        ctx.save();
        ctx.font = "700 11px Inter, Segoe UI, sans-serif";
        ctx.textAlign = "center";
        ctx.fillStyle = "#ecfeff";
        chart.data.datasets.forEach((dataset, datasetIndex) => {
            const meta = chart.getDatasetMeta(datasetIndex);
            meta.data.forEach((element, index) => {
                const speedup = chart.data.speedups?.[index];
                if (!speedup) return;
                const position = element.tooltipPosition();
                ctx.fillText(`Sp ${Number(speedup).toFixed(2)}x`, position.x, position.y - 9);
            });
        });
        ctx.restore();
    },
};

async function loadBenchmarks() {
    if (!window.Chart) return;
    chartDefaults();
    try {
        const response = await fetch("/api/benchmark-results");
        const data = await response.json();
        const results = data.results || [];
        if (!results.length) return;

        const latestResolution = results[results.length - 1]?.resolution;
        const latest = results.filter(row => row.resolution === latestResolution);
        const cpuResults = latest.filter(row => !row.mode.includes("CUDA"));
        const labels = cpuResults.map(row => row.mode.replace("CPU ", "").replace("Baseline", "Sequential"));
        const times = cpuResults.map(row => row.execution_time_ms);
        const throughput = cpuResults.map(row => row.throughput_mpps);
        const gpu = latest.find(row => row.mode.includes("CUDA"));
        const sequential = latest.find(row => row.mode.includes("Sequential"))?.execution_time_ms || times[0] || 0;
        const fastestCpu = Math.min(...times);
        const peakSpeed = sequential > 0 ? sequential / fastestCpu : 0;
        const allSpeedups = latest
            .map(row => typeof row.speedup === "number" ? row.speedup : null)
            .filter(value => value !== null);
        const peakAchieved = Math.max(1, peakSpeed, ...allSpeedups);
        const savings = sequential > 0 && fastestCpu > 0 ? ((sequential - fastestCpu) / sequential) * 100 : 0;
        const speedups = cpuResults.map(row =>
            typeof row.speedup === "number"
                ? row.speedup
                : (sequential && row.execution_time_ms > 0 ? Math.round((sequential / row.execution_time_ms) * 100) / 100 : 1)
        );

        document.querySelector("#peakCpu").textContent = `${peakAchieved.toFixed(2)}x`;
        document.querySelector("#gpuTime").textContent = gpu ? `${gpu.throughput_mpps.toFixed(1)} MP/s` : "—";
        document.querySelector("#totalSavings").textContent = `${Math.max(0, savings).toFixed(1)}%`;
        document.querySelector("#sequentialSpeedup").textContent = "1.00x (Baseline)";
        const openmpLatest = [...cpuResults].reverse().find(row => row.mode.includes("OpenMP"));
        if (openmpLatest?.speedup) {
            setSpeedupBadge("#openmpSpeedup", openmpLatest.speedup);
        }

        if (!charts.time) {
            charts.time = new Chart(document.querySelector("#timeChart"), {
                type: "bar",
                data: { labels, speedups, datasets: [{ label: "Execution Time (ms)", data: times, backgroundColor: ["rgba(34,211,238,.85)", "rgba(96,165,250,.85)", "rgba(129,140,248,.85)", "rgba(167,139,250,.85)", "rgba(52,211,153,.85)"], borderRadius: 7, borderSkipped: false }] },
                plugins: [speedupDataLabelPlugin],
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: { label: context => `${context.formattedValue} ms · Sp ${Number(speedups[context.dataIndex] || 1).toFixed(2)}x` } } }, scales: { y: { beginAtZero: true, title: { display: true, text: "milliseconds" } }, x: { grid: { display: false } } } }
            });
        } else {
            charts.time.data.labels = labels;
            charts.time.data.speedups = speedups;
            charts.time.data.datasets[0].data = times;
            charts.time.update();
        }

        const threadLabels = cpuResults.map(row => {
            const match = row.mode.match(/(\d+)\s+Threads?/);
            return match ? `${match[1]}T` : "1T";
        });
        if (!charts.throughput) {
            charts.throughput = new Chart(document.querySelector("#throughputChart"), {
                type: "line",
                data: { labels: threadLabels, speedups, datasets: [{ label: "CPU Throughput (MP/s)", data: throughput, borderColor: "#22d3ee", backgroundColor: "rgba(34,211,238,.12)", fill: true, tension: .35, pointRadius: 5, pointBackgroundColor: "#22d3ee" }, ...(gpu ? [{ label: "CUDA GPU", data: threadLabels.map((_, index) => index === threadLabels.length - 1 ? gpu.throughput_mpps : null), borderColor: "#a78bfa", backgroundColor: "rgba(167,139,250,.12)", pointBackgroundColor: "#a78bfa", spanGaps: false }] : [])] },
                plugins: [speedupDataLabelPlugin],
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { boxWidth: 10 } } }, scales: { y: { beginAtZero: true, title: { display: true, text: "Megapixels/sec" } }, x: { title: { display: true, text: "Thread configuration" }, grid: { display: false } } } }
            });
        } else {
            charts.throughput.data.labels = threadLabels;
            charts.throughput.data.speedups = speedups;
            charts.throughput.data.datasets[0].data = throughput;
            if (charts.throughput.data.datasets[1]) charts.throughput.data.datasets[1].data = threadLabels.map((_, index) => index === threadLabels.length - 1 && gpu ? gpu.throughput_mpps : null);
            charts.throughput.update();
        }
    } catch (_) {
        showToast("Unable to load benchmark analytics.");
    }
}

window.addEventListener("resize", () => updateSplit());
loadBenchmarks();
