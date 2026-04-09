/**
 * YouTube Sentiment Analyzer — Frontend Logic
 * Handles form submission, API calls, Chart.js rendering,
 * comments table with pagination/search/filter.
 */

// ============================================
// State
// ============================================
let allComments = [];
let filteredComments = [];
let currentPage = 1;
const PAGE_SIZE = 25;
let doughnutChart = null;
let histogramChart = null;
let comparisonChart = null;

// ============================================
// DOM References
// ============================================
const form = document.getElementById("analyze-form");
const btnAnalyze = document.getElementById("btn-analyze");
const btnText = btnAnalyze.querySelector(".btn-text");
const btnLoader = btnAnalyze.querySelector(".btn-loader");
const errorCard = document.getElementById("error-card");
const errorMessage = document.getElementById("error-message");
const resultsSection = document.getElementById("results-section");
const commentSlider = document.getElementById("max-comments");
const commentCountDisplay = document.getElementById("comment-count-display");
const commentSearch = document.getElementById("comment-search");
const commentFilter = document.getElementById("comment-filter");
const btnPrev = document.getElementById("btn-prev");
const btnNext = document.getElementById("btn-next");

// ============================================
// Initialization
// ============================================
document.addEventListener("DOMContentLoaded", () => {
    // Slider display
    commentSlider.addEventListener("input", () => {
        commentCountDisplay.textContent = commentSlider.value;
    });

    // Form submission
    form.addEventListener("submit", handleSubmit);

    // Search / filter
    commentSearch.addEventListener("input", debounce(applyFilters, 250));
    commentFilter.addEventListener("change", applyFilters);

    // Pagination
    btnPrev.addEventListener("click", () => { currentPage--; renderCommentsPage(); });
    btnNext.addEventListener("click", () => { currentPage++; renderCommentsPage(); });

    // Load model info
    fetchModelInfo();
});

// ============================================
// Utilities
// ============================================
function debounce(fn, ms) {
    let timer;
    return (...args) => {
        clearTimeout(timer);
        timer = setTimeout(() => fn(...args), ms);
    };
}

function formatNumber(n) {
    if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
    return n.toLocaleString();
}

function animateCounter(el, target, duration = 800) {
    const start = 0;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        const current = Math.round(start + (target - start) * eased);

        if (typeof target === "number" && Number.isInteger(target)) {
            el.textContent = current.toLocaleString();
        } else {
            el.textContent = (start + (target - start) * eased).toFixed(2);
        }

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

function getBadgeClass(label) {
    switch (label) {
        case "positive": return "badge-positive";
        case "neutral": return "badge-neutral";
        case "negative": return "badge-negative";
        default: return "badge-na";
    }
}

function getScoreColor(compound) {
    if (compound >= 0.05) return "var(--accent-green)";
    if (compound <= -0.05) return "var(--accent-red)";
    return "var(--accent-yellow)";
}

// ============================================
// Fetch Model Info
// ============================================
async function fetchModelInfo() {
    try {
        const res = await fetch("/api/model-info");
        const data = await res.json();

        const badgeModel = document.getElementById("badge-model");
        if (data.ml_model_available) {
            badgeModel.innerHTML = `<span class="badge-dot"></span> Model: ${data.model_type || "Loaded"}`;
        } else {
            badgeModel.innerHTML = `<span class="badge-dot dot-inactive"></span> Model: Not Trained`;
        }
    } catch {
        // Silently ignore — model info is not critical
    }
}

// ============================================
// Form Handler
// ============================================
async function handleSubmit(e) {
    e.preventDefault();
    hideError();
    setLoading(true);

    const payload = {
        video_url: document.getElementById("video-url").value.trim(),
        api_key: document.getElementById("api-key").value.trim(),
        max_comments: parseInt(commentSlider.value),
    };

    try {
        const res = await fetch("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        const data = await res.json();

        if (!res.ok) {
            showError(data.error || "An error occurred");
            return;
        }

        renderResults(data);
    } catch (err) {
        showError("Failed to connect to the server. Is the Flask app running?");
    } finally {
        setLoading(false);
    }
}

function setLoading(loading) {
    btnAnalyze.disabled = loading;
    btnText.style.display = loading ? "none" : "inline";
    btnLoader.style.display = loading ? "inline-flex" : "none";
}

function showError(msg) {
    errorCard.style.display = "flex";
    errorMessage.textContent = msg;
}

function hideError() {
    errorCard.style.display = "none";
}

// ============================================
// Render Results
// ============================================
function renderResults(data) {
    resultsSection.style.display = "block";

    renderVideoInfo(data.video);
    renderStats(data.stats);
    renderDoughnutChart(data.stats);
    renderHistogram(data.histogram);
    renderModelComparison(data);
    renderModelInfo(data);
    renderTopComments(data.top_positive, data.top_negative);

    allComments = data.comments;
    currentPage = 1;
    applyFilters();

    // Trigger animations
    document.querySelectorAll("#results-section > *").forEach((el, i) => {
        el.classList.add("animate-in", `delay-${Math.min(i + 1, 5)}`);
    });

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ============================================
// Video Info
// ============================================
function renderVideoInfo(video) {
    document.getElementById("video-thumbnail").src = video.thumbnail;
    document.getElementById("video-title").textContent = video.title;
    document.getElementById("video-channel").textContent = video.channel;
    document.getElementById("video-views").textContent = formatNumber(video.view_count) + " views";
    document.getElementById("video-likes").textContent = formatNumber(video.like_count) + " likes";
    document.getElementById("video-comments-count").textContent = formatNumber(video.comment_count) + " comments";
}

// ============================================
// Stats Cards
// ============================================
function renderStats(stats) {
    const total = stats.total_comments || 1;
    const v = stats.vader;

    // Animate counters
    animateCounter(document.getElementById("stat-positive"), v.positive);
    animateCounter(document.getElementById("stat-neutral"), v.neutral);
    animateCounter(document.getElementById("stat-negative"), v.negative);
    animateCounter(document.getElementById("stat-avg-compound"), v.avg_compound);

    // Animate bars
    setTimeout(() => {
        document.getElementById("bar-positive").style.width = `${(v.positive / total) * 100}%`;
        document.getElementById("bar-neutral").style.width = `${(v.neutral / total) * 100}%`;
        document.getElementById("bar-negative").style.width = `${(v.negative / total) * 100}%`;
        const avgPct = ((v.avg_compound + 1) / 2) * 100;
        document.getElementById("bar-avg").style.width = `${avgPct}%`;
    }, 100);
}

// ============================================
// Doughnut Chart
// ============================================
function renderDoughnutChart(stats) {
    const v = stats.vader;
    const ctx = document.getElementById("chart-doughnut").getContext("2d");

    if (doughnutChart) doughnutChart.destroy();

    doughnutChart = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["Positive", "Neutral", "Negative"],
            datasets: [{
                data: [v.positive, v.neutral, v.negative],
                backgroundColor: [
                    "rgba(0, 230, 118, 0.8)",
                    "rgba(255, 202, 40, 0.8)",
                    "rgba(255, 82, 82, 0.8)",
                ],
                borderColor: [
                    "rgba(0, 230, 118, 1)",
                    "rgba(255, 202, 40, 1)",
                    "rgba(255, 82, 82, 1)",
                ],
                borderWidth: 2,
                hoverOffset: 12,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: "65%",
            plugins: {
                legend: {
                    position: "bottom",
                    labels: {
                        color: "#8b8ba3",
                        padding: 20,
                        font: { family: "Inter", size: 12, weight: 500 },
                        usePointStyle: true,
                        pointStyleWidth: 10,
                    },
                },
                tooltip: {
                    backgroundColor: "rgba(18, 18, 30, 0.95)",
                    titleColor: "#e8e8ef",
                    bodyColor: "#8b8ba3",
                    borderColor: "rgba(255,255,255,0.06)",
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    titleFont: { family: "Inter", weight: 600 },
                    bodyFont: { family: "Inter" },
                },
            },
            animation: {
                animateRotate: true,
                duration: 1200,
            },
        },
    });
}

// ============================================
// Histogram Chart
// ============================================
function renderHistogram(histogram) {
    const ctx = document.getElementById("chart-histogram").getContext("2d");

    if (histogramChart) histogramChart.destroy();

    const colors = histogram.bins.map((b) => {
        const val = parseFloat(b);
        if (val >= 0.05) return "rgba(0, 230, 118, 0.7)";
        if (val <= -0.1) return "rgba(255, 82, 82, 0.7)";
        return "rgba(255, 202, 40, 0.7)";
    });

    histogramChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: histogram.bins,
            datasets: [{
                label: "Comments",
                data: histogram.counts,
                backgroundColor: colors,
                borderColor: colors.map(c => c.replace("0.7", "1")),
                borderWidth: 1,
                borderRadius: 6,
                barPercentage: 0.85,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "rgba(18, 18, 30, 0.95)",
                    titleColor: "#e8e8ef",
                    bodyColor: "#8b8ba3",
                    borderColor: "rgba(255,255,255,0.06)",
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    titleFont: { family: "Inter", weight: 600 },
                    bodyFont: { family: "Inter" },
                },
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: "Compound Score",
                        color: "#5a5a72",
                        font: { family: "Inter", size: 11 },
                    },
                    ticks: { color: "#5a5a72", font: { family: "Inter", size: 10 } },
                    grid: { color: "rgba(255,255,255,0.03)" },
                },
                y: {
                    title: {
                        display: true,
                        text: "Count",
                        color: "#5a5a72",
                        font: { family: "Inter", size: 11 },
                    },
                    ticks: { color: "#5a5a72", font: { family: "Inter", size: 10 } },
                    grid: { color: "rgba(255,255,255,0.03)" },
                    beginAtZero: true,
                },
            },
            animation: { duration: 1000 },
        },
    });
}

// ============================================
// Model Comparison Chart
// ============================================
function renderModelComparison(data) {
    const card = document.getElementById("chart-comparison-card");

    if (!data.ml_model_available || !data.stats.ml_model) {
        card.style.display = "none";
        return;
    }

    card.style.display = "block";
    const ctx = document.getElementById("chart-comparison").getContext("2d");

    if (comparisonChart) comparisonChart.destroy();

    const v = data.stats.vader;
    const m = data.stats.ml_model;

    comparisonChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["Positive", "Neutral", "Negative"],
            datasets: [
                {
                    label: "VADER",
                    data: [v.positive, v.neutral, v.negative],
                    backgroundColor: "rgba(124, 77, 255, 0.6)",
                    borderColor: "rgba(124, 77, 255, 1)",
                    borderWidth: 1,
                    borderRadius: 6,
                },
                {
                    label: data.model_metrics?.model_type?.toUpperCase() || "ML Model",
                    data: [m.positive, m.neutral, m.negative],
                    backgroundColor: "rgba(68, 138, 255, 0.6)",
                    borderColor: "rgba(68, 138, 255, 1)",
                    borderWidth: 1,
                    borderRadius: 6,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: {
                        color: "#8b8ba3",
                        padding: 20,
                        font: { family: "Inter", size: 12, weight: 500 },
                        usePointStyle: true,
                    },
                },
                tooltip: {
                    backgroundColor: "rgba(18, 18, 30, 0.95)",
                    titleColor: "#e8e8ef",
                    bodyColor: "#8b8ba3",
                    borderColor: "rgba(255,255,255,0.06)",
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                },
            },
            scales: {
                x: {
                    ticks: { color: "#5a5a72", font: { family: "Inter", size: 11 } },
                    grid: { color: "rgba(255,255,255,0.03)" },
                },
                y: {
                    ticks: { color: "#5a5a72", font: { family: "Inter", size: 11 } },
                    grid: { color: "rgba(255,255,255,0.03)" },
                    beginAtZero: true,
                },
            },
            animation: { duration: 1000 },
        },
    });
}

// ============================================
// Model Performance Card
// ============================================
function renderModelInfo(data) {
    const card = document.getElementById("model-card");
    const grid = document.getElementById("model-metrics-grid");

    if (!data.model_metrics) {
        card.style.display = "none";
        return;
    }

    card.style.display = "block";
    const m = data.model_metrics;

    grid.innerHTML = `
        <div class="metric-item">
            <div class="metric-value">${(m.accuracy * 100).toFixed(1)}%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        <div class="metric-item">
            <div class="metric-value">${(m.precision * 100).toFixed(1)}%</div>
            <div class="metric-label">Precision</div>
        </div>
        <div class="metric-item">
            <div class="metric-value">${(m.recall * 100).toFixed(1)}%</div>
            <div class="metric-label">Recall</div>
        </div>
        <div class="metric-item">
            <div class="metric-value">${(m.f1_score * 100).toFixed(1)}%</div>
            <div class="metric-label">F1 Score</div>
        </div>
    `;

    // Update header badge
    const badgeModel = document.getElementById("badge-model");
    badgeModel.innerHTML = `<span class="badge-dot"></span> Model: ${m.model_type} (${(m.accuracy * 100).toFixed(0)}%)`;
}

// ============================================
// Top Comments
// ============================================
function renderTopComments(topPositive, topNegative) {
    const posList = document.getElementById("top-positive-list");
    const negList = document.getElementById("top-negative-list");

    posList.innerHTML = topPositive.map(c => `
        <div class="top-comment-item">
            <div class="top-comment-text">${escapeHtml(c.text)}</div>
            <div class="top-comment-meta">
                <span>${escapeHtml(c.author)}</span>
                <span class="top-comment-score score-positive">${c.sentiment.vader.compound.toFixed(3)}</span>
            </div>
        </div>
    `).join("");

    negList.innerHTML = topNegative.map(c => `
        <div class="top-comment-item">
            <div class="top-comment-text">${escapeHtml(c.text)}</div>
            <div class="top-comment-meta">
                <span>${escapeHtml(c.author)}</span>
                <span class="top-comment-score score-negative">${c.sentiment.vader.compound.toFixed(3)}</span>
            </div>
        </div>
    `).join("");
}

// ============================================
// Comments Table
// ============================================
function applyFilters() {
    const searchTerm = commentSearch.value.toLowerCase();
    const filterVal = commentFilter.value;

    filteredComments = allComments.filter(c => {
        const matchSearch = !searchTerm ||
            c.text.toLowerCase().includes(searchTerm) ||
            c.author.toLowerCase().includes(searchTerm);
        const matchFilter = filterVal === "all" || c.sentiment.vader.label === filterVal;
        return matchSearch && matchFilter;
    });

    currentPage = 1;
    renderCommentsPage();
}

function renderCommentsPage() {
    const tbody = document.getElementById("comments-tbody");
    const totalPages = Math.max(1, Math.ceil(filteredComments.length / PAGE_SIZE));

    if (currentPage > totalPages) currentPage = totalPages;
    if (currentPage < 1) currentPage = 1;

    const start = (currentPage - 1) * PAGE_SIZE;
    const end = Math.min(start + PAGE_SIZE, filteredComments.length);
    const pageComments = filteredComments.slice(start, end);

    tbody.innerHTML = pageComments.map(c => {
        const v = c.sentiment.vader;
        const ml = c.sentiment.ml_model;

        return `
            <tr>
                <td class="col-author">${escapeHtml(c.author)}</td>
                <td class="col-text">${escapeHtml(truncate(c.text, 200))}</td>
                <td><span class="sentiment-badge ${getBadgeClass(v.label)}">${v.label}</span></td>
                <td>${ml
                    ? `<span class="sentiment-badge ${getBadgeClass(ml.label)}">${ml.label}</span>`
                    : `<span class="sentiment-badge badge-na">N/A</span>`
                }</td>
                <td class="col-score" style="color: ${getScoreColor(v.compound)}">${v.compound.toFixed(3)}</td>
                <td>${c.like_count}</td>
            </tr>
        `;
    }).join("");

    // Pagination controls
    document.getElementById("page-info").textContent = `Page ${currentPage} of ${totalPages}`;
    btnPrev.disabled = currentPage <= 1;
    btnNext.disabled = currentPage >= totalPages;

    document.getElementById("comment-total").textContent = `(${filteredComments.length})`;
}

function truncate(str, len) {
    if (!str) return "";
    if (str.length <= len) return str;
    return str.substring(0, len) + "…";
}

function escapeHtml(str) {
    if (!str) return "";
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}
