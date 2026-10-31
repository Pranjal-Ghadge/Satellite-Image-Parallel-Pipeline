const state = { uploadedFilename: null };

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
    if (!data || !data.detail) return fallback;
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail)) {
        return data.detail.map(item => item.msg || JSON.stringify(item)).join("; ");
    }
    return JSON.stringify(data.detail);
}

function setImage(elementId, downloadId, url, filename) {
    const image = document.querySelector(elementId);
    const download = document.querySelector(downloadId);
    const frame = image.parentElement;
    const placeholder = frame.querySelector(".placeholder");
    image.src = url || "";
    image.style.display = url ? "block" : "none";
    if (placeholder) placeholder.style.display = url ? "none" : "";
    if (url) {
        download.hidden = false;
        download.href = url;
        download.download = filename;
    } else {
        download.hidden = true;
    }
}

async function uploadFile(file) {
    if (!file) return;
    const body = new FormData();
    body.append("file", file);

    elements.runButton.disabled = true;
    try {
        const response = await fetch("/api/upload", { method: "POST", body });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Upload failed");
        state.uploadedFilename = data.filename;
        setImage("#inputImage", "#downloadInput", data.url, data.filename);
        showToast("Image uploaded successfully");
    } catch (error) {
        showToast(error.message);
    } finally {
        elements.runButton.disabled = false;
    }
}

elements.uploadZone.addEventListener("click", () => elements.imageInput.click());
elements.imageInput.addEventListener("change", () => uploadFile(elements.imageInput.files[0]));
elements.uploadZone.addEventListener("dragover", event => {
    event.preventDefault();
    elements.uploadZone.classList.add("dragging");
});
elements.uploadZone.addEventListener("dragleave", () => elements.uploadZone.classList.remove("dragging"));
elements.uploadZone.addEventListener("drop", event => {
    event.preventDefault();
    elements.uploadZone.classList.remove("dragging");
    uploadFile(event.dataTransfer.files[0]);
});

elements.cuda.addEventListener("change", () => {
    const cudaPlaceholder = document.getElementById("cudaPlaceholder");
    if (!elements.cuda.checked) {
        const cudaImage = document.getElementById("cuda-img");
        cudaImage.removeAttribute("src");
        cudaImage.style.display = "none";
        cudaPlaceholder.style.display = "";
        cudaPlaceholder.textContent = "CUDA Disabled";
        document.querySelector("#downloadCuda").hidden = true;
    } else {
        cudaPlaceholder.textContent = "Awaiting CUDA Run";
    }
});

elements.runButton.addEventListener("click", async () => {
    if (!state.uploadedFilename) {
        showToast("Upload an image before running the pipeline.");
        return;
    }

    elements.runButton.disabled = true;
    elements.runButton.textContent = "Processing...";
    try {
        const response = await fetch("/api/run-pipeline", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                input_filename: state.uploadedFilename,
                resolution: elements.resolution.value,
                threads: Number(elements.threads.value),
                enable_cuda: elements.cuda.checked,
            }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(getApiError(data, "Pipeline failed"));

        setImage("#inputImage", "#downloadInput", data.prepared_input, "input_prepared.jpg");
        setImage("#sequentialImage", "#downloadSequential", data.outputs.sequential, "sequential_edge.jpg");
        setImage("#openmpImage", "#downloadOpenmp", data.outputs.openmp, "openmp_edge.jpg");

        const cudaImage = document.getElementById("cuda-img");
        const cudaPlaceholder = document.getElementById("cudaPlaceholder");
        if (data.cuda_image_url) {
            cudaPlaceholder.style.display = "none";
            cudaImage.style.display = "block";
            cudaImage.src = data.cuda_image_url + "?t=" + new Date().getTime();
            const cudaDownload = document.querySelector("#downloadCuda");
            cudaDownload.hidden = false;
            cudaDownload.href = data.cuda_image_url;
            cudaDownload.download = "cuda_edge.jpg";
        } else if (data.cuda_requested) {
            cudaImage.removeAttribute("src");
            cudaImage.style.display = "none";
            cudaPlaceholder.style.display = "";
            cudaPlaceholder.textContent = data.cuda_status === "unavailable_build"
                ? "CUDA Not Built"
                : "CUDA Unavailable";
        } else {
            cudaImage.removeAttribute("src");
            cudaImage.style.display = "none";
            cudaPlaceholder.style.display = "";
            cudaPlaceholder.textContent = "CUDA Disabled";
        }

        showToast(data.cuda_image_url ? "Pipeline and CUDA execution completed" : "Pipeline completed");
        await loadBenchmarks();
    } catch (error) {
        showToast(error.message);
    } finally {
        elements.runButton.disabled = false;
        elements.runButton.textContent = "Run Benchmark";
    }
});

function shortenMode(mode) {
    if (mode.includes("Sequential")) return "Sequential";
    if (mode.includes("OpenMP")) {
        const match = mode.match(/(\d+)\s+Threads?/);
        return match ? `OpenMP ${match[1]}` : "OpenMP";
    }
    if (mode.includes("CUDA")) return "CUDA";
    return mode;
}

function drawBarChart(canvas, labels, values, colors) {
    const ctx = canvas.getContext("2d");
    const ratio = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * ratio;
    canvas.height = rect.height * ratio;
    ctx.scale(ratio, ratio);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const width = rect.width;
    const height = rect.height;
    const left = 38;
    const bottom = 28;
    const chartWidth = width - left - 12;
    const chartHeight = height - bottom - 10;
    const max = Math.max(...values, 1) * 1.12;

    ctx.strokeStyle = "rgba(148,163,184,.14)";
    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px Segoe UI";
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
        const y = 10 + chartHeight * i / 4;
        const value = max * (1 - i / 4);
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(width - 6, y);
        ctx.stroke();
        ctx.fillText(value.toFixed(value >= 10 ? 0 : 1), left - 5, y + 3);
    }

    const slot = chartWidth / labels.length;
    const barWidth = Math.min(46, slot * 0.62);
    labels.forEach((label, index) => {
        const barHeight = values[index] / max * chartHeight;
        const x = left + slot * index + (slot - barWidth) / 2;
        const y = height - bottom - barHeight;
        ctx.fillStyle = colors[index % colors.length];
        ctx.fillRect(x, y, barWidth, barHeight);
        ctx.fillStyle = "#94a3b8";
        ctx.textAlign = "center";
        ctx.fillText(label, x + barWidth / 2, height - 8);
    });
}

async function loadBenchmarks() {
    try {
        const response = await fetch("/api/benchmark-results");
        const data = await response.json();
        const results = data.results || [];
        if (!results.length) return;

        const latestResolution = results[results.length - 1]?.resolution;
        const latest = results.filter(row => row.resolution === latestResolution);
        const labels = latest.map(row => shortenMode(row.mode));
        const times = latest.map(row => row.execution_time_ms);
        const throughput = latest.map(row => row.throughput_mpps);
        const sequential = times[0] || 0;
        const speeds = times.map(time => sequential && time > 0 ? sequential / time : 0);
        const colors = ["#22d3ee", "#60a5fa", "#818cf8", "#a78bfa", "#34d399", "#f472b6"];

        drawBarChart(document.querySelector("#timeChart"), labels, times, colors);
        drawBarChart(document.querySelector("#throughputChart"), labels, throughput, colors);
        drawBarChart(document.querySelector("#speedupChart"), labels, speeds, colors);

        const peakSpeed = Math.max(...speeds.filter(value => Number.isFinite(value)));
        const gpu = latest.find(row => row.mode.includes("CUDA"));
        const totalThroughput = throughput.reduce((sum, value) => sum + value, 0);
        document.querySelector("#peakCpu").textContent = `${peakSpeed.toFixed(2)}×`;
        document.querySelector("#gpuTime").textContent = gpu ? `${gpu.execution_time_ms.toFixed(2)} ms` : "—";
        document.querySelector("#totalThroughput").textContent = `${totalThroughput.toFixed(1)} MP/s`;
    } catch (error) {
        showToast("Unable to load benchmark analytics.");
    }
}

window.addEventListener("resize", loadBenchmarks);
loadBenchmarks();
