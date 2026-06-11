// Fall Detection System - Main JavaScript

// Global variables
let selectedFile = null;
let currentResults = null;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const selectedFileDiv = document.getElementById('selectedFile');
const fileName = document.getElementById('fileName');
const removeFileBtn = document.getElementById('removeFile');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const resultsSection = document.getElementById('resultsSection');
const analyzeAnotherBtn = document.getElementById('analyzeAnotherBtn');
const downloadReportBtn = document.getElementById('downloadReport');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkSystemStatus();
});

// Setup Event Listeners
function setupEventListeners() {
    // Upload area click
    if (uploadArea) uploadArea.addEventListener('click', () => fileInput && fileInput.click());

    // File input change
    if (fileInput) fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    if (uploadArea) {
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', handleDrop);
    }

    // Remove file button
    if (removeFileBtn) removeFileBtn.addEventListener('click', removeFile);

    // Analyze button
    if (analyzeBtn) analyzeBtn.addEventListener('click', analyzeFile);

    // Analyze another button
    if (analyzeAnotherBtn) analyzeAnotherBtn.addEventListener('click', resetForm);

    // Download report button
    if (downloadReportBtn) downloadReportBtn.addEventListener('click', downloadReport);
}

// Check system status
async function checkSystemStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        if (data.status === 'online' && data.model_loaded) {
            updateStatusIndicator('online', 'System Ready');
        } else {
            updateStatusIndicator('warning', 'Model Not Loaded');
        }
    } catch (error) {
        updateStatusIndicator('offline', 'System Offline');
    }
}

// Update status indicator
function updateStatusIndicator(status, text) {
    const indicator = document.getElementById('statusIndicator');
    const statusDot = indicator.querySelector('.status-dot');
    const statusText = indicator.querySelector('.status-text');

    statusText.textContent = text;

    if (status === 'online') {
        indicator.style.borderColor = 'var(--success-color)';
        statusDot.style.background = 'var(--success-color)';
        statusText.style.color = 'var(--success-color)';
    } else if (status === 'warning') {
        indicator.style.borderColor = 'var(--warning-color)';
        statusDot.style.background = 'var(--warning-color)';
        statusText.style.color = 'var(--warning-color)';
    } else {
        indicator.style.borderColor = 'var(--danger-color)';
        statusDot.style.background = 'var(--danger-color)';
        statusText.style.color = 'var(--danger-color)';
    }
}

// Handle file select
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        setSelectedFile(file);
    }
}

// Handle drag over
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
}

// Handle drag leave
function handleDragLeave(e) {
    uploadArea.classList.remove('drag-over');
}

// Handle drop
function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (file.name.endsWith('.csv') || file.name.endsWith('.txt')) {
            setSelectedFile(file);
        } else {
            showError('Please upload a CSV or TXT file');
        }
    }
}

// Set selected file
function setSelectedFile(file) {
    selectedFile = file;
    fileName.textContent = file.name;
    selectedFileDiv.style.display = 'flex';
    uploadArea.style.display = 'none';
    analyzeBtn.disabled = false;
}

// Remove file
function removeFile(e) {
    e.stopPropagation();
    selectedFile = null;
    fileInput.value = '';
    selectedFileDiv.style.display = 'none';
    uploadArea.style.display = 'block';
    analyzeBtn.disabled = true;
}

// Analyze file
async function analyzeFile() {
    if (!selectedFile) {
        showError('No file selected');
        return;
    }

    // Show loading overlay
    loadingOverlay.style.display = 'flex';

    // Prepare form data
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.success) {
            currentResults = data;
            displayResults(data);
        } else {
            showError(data.error || 'Analysis failed');
        }
    } catch (error) {
        showError(`Error: ${error.message}`);
    } finally {
        loadingOverlay.style.display = 'none';
    }
}

// Display results
function displayResults(data) {
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    resultsSection.style.display = 'block';

    // Update final decision
    const finalDecisionBox = document.getElementById('finalDecisionBox');
    const resultIcon = document.getElementById('resultIcon');
    const resultTitle = document.getElementById('resultTitle');
    const resultDescription = document.getElementById('resultDescription');

    const isFall = data.final_decision.is_fall;
    const confidence = (data.final_decision.confidence * 100).toFixed(1);

    if (isFall) {
        finalDecisionBox.className = 'result-box fall-detected';
        resultIcon.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
        resultTitle.textContent = '⚠️ FALL DETECTED';
        resultDescription.textContent = `High confidence fall detection (${confidence}% probability). Immediate attention recommended.`;
    } else {
        finalDecisionBox.className = 'result-box no-fall';
        resultIcon.innerHTML = '<i class="fas fa-check-circle"></i>';
        resultTitle.textContent = '✓ No Fall Detected';
        resultDescription.textContent = `Activity classified as normal ADL (${(100 - confidence).toFixed(1)}% confidence).`;
    }

    // Update model prediction
    const modelPrediction = document.getElementById('modelPrediction');
    const modelConfidence = document.getElementById('modelConfidence');
    const modelProgress = document.getElementById('modelProgress');

    modelPrediction.textContent = data.model_prediction.is_fall_model ? 'Fall' : 'Normal ADL';
    modelPrediction.style.color = data.model_prediction.is_fall_model ? 'var(--danger-color)' : 'var(--success-color)';
    modelConfidence.textContent = `${(data.model_prediction.average_probability * 100).toFixed(1)}%`;
    modelProgress.style.width = `${data.model_prediction.average_probability * 100}%`;

    // Update threshold detection
    const thresholdPrediction = document.getElementById('thresholdPrediction');
    const peakAcceleration = document.getElementById('peakAcceleration');
    const threshold = document.getElementById('threshold');

    thresholdPrediction.textContent = data.threshold_detection.threshold_exceeded ? 'Exceeded' : 'Normal';
    thresholdPrediction.style.color = data.threshold_detection.threshold_exceeded ? 'var(--danger-color)' : 'var(--success-color)';
    peakAcceleration.textContent = `${data.threshold_detection.peak_acceleration.toFixed(2)}g`;
    threshold.textContent = `${data.threshold_detection.threshold}g`;

    // Update data statistics
    const dataDuration = document.getElementById('dataDuration');
    const dataSamples = document.getElementById('dataSamples');
    const dataWindows = document.getElementById('dataWindows');

    dataDuration.textContent = `${data.duration_seconds.toFixed(2)}s`;
    dataSamples.textContent = data.data_length.toLocaleString();
    dataWindows.textContent = data.model_prediction.num_windows.toLocaleString();

    // Update configuration
    const validationMode = document.getElementById('validationMode');
    validationMode.textContent = data.final_decision.validation_mode;
}

// Reset form
function resetForm() {
    removeFile(new Event('click'));
    resultsSection.style.display = 'none';
    currentResults = null;
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Download report
function downloadReport() {
    if (!currentResults) return;

    const report = generateReport(currentResults);
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `fall_detection_report_${new Date().toISOString().slice(0, 10)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Generate report
function generateReport(data) {
    const report = `
=============================================================================
FALL DETECTION SYSTEM - ANALYSIS REPORT
=============================================================================

Report Generated: ${new Date().toLocaleString()}
File Name: ${data.filename}

=============================================================================
FINAL DECISION
=============================================================================

Result: ${data.final_decision.is_fall ? '⚠️ FALL DETECTED' : '✓ No Fall Detected'}
Confidence: ${(data.final_decision.confidence * 100).toFixed(1)}%
Validation Mode: ${data.final_decision.validation_mode}

=============================================================================
MODEL PREDICTION (CNN-LSTM)
=============================================================================

Classification: ${data.model_prediction.is_fall_model ? 'Fall' : 'Normal ADL'}
Average Probability: ${(data.model_prediction.average_probability * 100).toFixed(2)}%
Max Probability: ${(data.model_prediction.max_probability * 100).toFixed(2)}%
Min Probability: ${(data.model_prediction.min_probability * 100).toFixed(2)}%
Number of Windows: ${data.model_prediction.num_windows}

=============================================================================
THRESHOLD-BASED DETECTION
=============================================================================

Status: ${data.threshold_detection.threshold_exceeded ? 'Threshold Exceeded' : 'Normal'}
Peak Acceleration: ${data.threshold_detection.peak_acceleration.toFixed(3)}g
Threshold: ${data.threshold_detection.threshold}g

=============================================================================
DATA STATISTICS
=============================================================================

Total Samples: ${data.data_length}
Duration: ${data.duration_seconds.toFixed(2)} seconds
Sampling Rate: 200 Hz
Window Size: 200 samples (1 second)

=============================================================================
SYSTEM INFORMATION
=============================================================================

Model: CNN-LSTM Hybrid Architecture
Training Dataset: SisFall (104,038 samples)
Model Parameters: 80,321
Framework: TensorFlow 2.19 + Keras 3.12

=============================================================================
END OF REPORT
=============================================================================
`;

    return report.trim();
}

// Show error
function showError(message) {
    alert(`Error: ${message}`);
}

// Utility: Format number with commas
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

// =============================================================================
// VIDEO UPLOAD FUNCTIONALITY
// =============================================================================

// Video-related global variables
let selectedVideo = null;

// Video DOM Elements
const videoUploadArea = document.getElementById('videoUploadArea');
const videoFileInput = document.getElementById('videoFileInput');
const selectedVideoFileDiv = document.getElementById('selectedVideoFile');
const videoFileName = document.getElementById('videoFileName');
const removeVideoFileBtn = document.getElementById('removeVideoFile');
const analyzeVideoBtn = document.getElementById('analyzeVideoBtn');
const videoLoadingOverlay = document.getElementById('videoLoadingOverlay');
const videoResultsSection = document.getElementById('videoResultsSection');
const analyzeAnotherVideoBtn = document.getElementById('analyzeAnotherVideoBtn');
const downloadVideoReportBtn = document.getElementById('downloadVideoReport');

// Tab Navigation
function setupTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active class from all tabs
            tabBtns.forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });

            // Add active class to clicked tab
            btn.classList.add('active');
            const tabId = btn.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });
}

// Setup Video Event Listeners
function setupVideoEventListeners() {
    // Upload area click
    if (videoUploadArea) {
        videoUploadArea.addEventListener('click', () => videoFileInput.click());

        // Drag and drop for video
        videoUploadArea.addEventListener('dragover', handleVideoDragOver);
        videoUploadArea.addEventListener('dragleave', handleVideoDragLeave);
        videoUploadArea.addEventListener('drop', handleVideoDrop);
    }

    // Video file input change
    if (videoFileInput) {
        videoFileInput.addEventListener('change', handleVideoFileSelect);
    }

    // Remove video file button
    if (removeVideoFileBtn) {
        removeVideoFileBtn.addEventListener('click', removeVideo);
    }

    // Analyze video button
    if (analyzeVideoBtn) {
        analyzeVideoBtn.addEventListener('click', analyzeVideo);
    }

    // Analyze another video button
    if (analyzeAnotherVideoBtn) {
        analyzeAnotherVideoBtn.addEventListener('click', resetVideoForm);
    }

    // Download video report button
    if (downloadVideoReportBtn) {
        downloadVideoReportBtn.addEventListener('click', downloadVideoReport);
    }
}

// Handle video file select
function handleVideoFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        setSelectedVideo(file);
    }
}

// Handle video drag over
function handleVideoDragOver(e) {
    e.preventDefault();
    videoUploadArea.classList.add('dragover');
}

// Handle video drag leave
function handleVideoDragLeave(e) {
    e.preventDefault();
    videoUploadArea.classList.remove('dragover');
}

// Handle video drop
function handleVideoDrop(e) {
    e.preventDefault();
    videoUploadArea.classList.remove('dragover');

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
        setSelectedVideo(file);
    }
}

// Set selected video
function setSelectedVideo(file) {
    selectedVideo = file;
    videoFileName.textContent = file.name;
    selectedVideoFileDiv.style.display = 'flex';
    analyzeVideoBtn.disabled = false;
}

// Remove video
function removeVideo(e) {
    e.stopPropagation();
    selectedVideo = null;
    videoFileInput.value = '';
    selectedVideoFileDiv.style.display = 'none';
    analyzeVideoBtn.disabled = true;
}

// Reset video form
function resetVideoForm() {
    removeVideo({ stopPropagation: () => { } });
    videoResultsSection.style.display = 'none';
}

// Analyze video
async function analyzeVideo() {
    if (!selectedVideo) return;

    // Show loading
    videoLoadingOverlay.style.display = 'flex';

    try {
        // Create form data
        const formData = new FormData();
        formData.append('video', selectedVideo);

        // Send request
        const response = await fetch('/upload_video', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Server error');
        }

        const data = await response.json();

        // Hide loading
        videoLoadingOverlay.style.display = 'none';

        // Display results
        displayVideoResults(data);

    } catch (error) {
        videoLoadingOverlay.style.display = 'none';
        showError('Failed to analyze video: ' + error.message);
    }
}

// Display video results
function displayVideoResults(data) {
    // Show results section
    videoResultsSection.style.display = 'block';
    videoResultsSection.scrollIntoView({ behavior: 'smooth' });

    // Update decision box
    const videoDecisionBox = document.getElementById('videoDecisionBox');
    const videoResultIcon = document.getElementById('videoResultIcon');
    const videoResultTitle = document.getElementById('videoResultTitle');
    const videoResultDescription = document.getElementById('videoResultDescription');

    if (data.fall_detected) {
        videoDecisionBox.className = 'result-box result-fall';
        videoResultIcon.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
        videoResultTitle.textContent = 'FALL DETECTED!';
        videoResultDescription.textContent = `${data.num_detections} fall event(s) detected in the video`;
    } else {
        videoDecisionBox.className = 'result-box result-safe';
        videoResultIcon.innerHTML = '<i class="fas fa-check-circle"></i>';
        videoResultTitle.textContent = 'No Falls Detected';
        videoResultDescription.textContent = 'The video analysis did not detect any fall events';
    }

    // Update video info
    document.getElementById('videoDuration').textContent = `${data.video_info.duration}s`;
    document.getElementById('videoResolution').textContent = `${data.video_info.width}x${data.video_info.height}`;
    document.getElementById('videoFPS').textContent = `${data.video_info.fps} FPS`;

    // Update analysis stats
    document.getElementById('fallsDetected').textContent = data.num_detections;
    document.getElementById('analyzedFrames').textContent = formatNumber(data.video_info.analyzed_frames);
    document.getElementById('totalFrames').textContent = formatNumber(data.video_info.total_frames);

    // Display screenshots if any falls detected
    const screenshotsContainer = document.getElementById('screenshotsContainer');
    const screenshotsGrid = document.getElementById('screenshotsGrid');

    if (data.fall_detected && data.screenshots.length > 0) {
        screenshotsContainer.style.display = 'block';
        screenshotsGrid.innerHTML = '';

        data.screenshots.forEach((screenshot, index) => {
            const detection = data.detections[index];
            const card = document.createElement('div');
            card.className = 'screenshot-card';
            card.innerHTML = `
                <img src="/static/${screenshot}" alt="Fall Detection ${index + 1}">
                <div class="screenshot-info">
                    <h4>Fall Detection #${index + 1}</h4>
                    <p>Time: ${detection.timestamp}s | Confidence: ${(detection.confidence * 100).toFixed(0)}%</p>
                </div>
            `;
            card.onclick = () => openLightbox(screenshot);
            screenshotsGrid.appendChild(card);
        });
    } else {
        screenshotsContainer.style.display = 'none';
    }

    // Display fall events timeline
    const fallEventsContainer = document.getElementById('fallEventsContainer');
    const fallEventsList = document.getElementById('fallEventsList');

    if (data.fall_detected && data.detections.length > 0) {
        fallEventsContainer.style.display = 'block';
        fallEventsList.innerHTML = '';

        data.detections.forEach((detection, index) => {
            const eventItem = document.createElement('div');
            eventItem.className = 'fall-event-item';
            eventItem.innerHTML = `
                <div class="fall-event-number">${index + 1}</div>
                <div class="fall-event-details">
                    <h4>Fall Detected at ${detection.timestamp}s</h4>
                    <p>Frame ${detection.frame}</p>
                </div>
                <div class="fall-event-confidence">${(detection.confidence * 100).toFixed(0)}%</div>
            `;
            fallEventsList.appendChild(eventItem);
        });
    } else {
        fallEventsContainer.style.display = 'none';
    }

    // Store results for report
    currentResults = data;
}

// Open lightbox for full-size screenshot
function openLightbox(screenshot) {
    const lightbox = document.createElement('div');
    lightbox.className = 'lightbox-overlay';
    lightbox.innerHTML = `
        <span class="lightbox-close">&times;</span>
        <img src="/static/${screenshot}" alt="Fall Detection">
    `;

    lightbox.onclick = () => lightbox.remove();
    document.body.appendChild(lightbox);
}

// Download video report
function downloadVideoReport() {
    if (!currentResults) return;

    const report = generateVideoReport(currentResults);
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `video_fall_detection_report_${new Date().getTime()}.txt`;
    a.click();

    URL.revokeObjectURL(url);
}

// Generate video report
function generateVideoReport(data) {
    const report = `
=============================================================================
VIDEO FALL DETECTION REPORT
=============================================================================

Timestamp: ${new Date().toLocaleString()}
Video File: ${selectedVideo?.name || 'Unknown'}

=============================================================================
DETECTION RESULTS
=============================================================================

Fall Detected: ${data.fall_detected ? 'YES' : 'NO'}
Number of Falls: ${data.num_detections}

${data.detections.map((d, i) => `
Fall #${i + 1}:
  - Time: ${d.timestamp}s
  - Frame: ${d.frame}
  - Confidence: ${(d.confidence * 100).toFixed(1)}%
`).join('')}

=============================================================================
VIDEO INFORMATION
=============================================================================

Duration: ${data.video_info.duration}s
Resolution: ${data.video_info.width}x${data.video_info.height}
Frame Rate: ${data.video_info.fps} FPS
Total Frames: ${data.video_info.total_frames}
Analyzed Frames: ${data.video_info.analyzed_frames}

=============================================================================
DETECTION METHOD
=============================================================================

Method: MediaPipe Pose Estimation + Motion Analysis
Analysis Rate: 10 FPS (every 3rd frame at 30 FPS)
Fall Criteria: Hip position drop + rapid downward velocity

=============================================================================
END OF REPORT
=============================================================================
`;

    return report.trim();
}

// Initialize everything when DOM loads
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    setupTabs();
    setupVideoEventListeners();
    setupTremorEventListeners();
    checkSystemStatus();
    checkTremorStatus();
    loadTremorDemoFiles();
});

// =============================================================================
// TREMOR DETECTION FUNCTIONALITY
// =============================================================================

let selectedTremorFile = null;
let tremorResults = null;

// DOM refs (grabbed lazily to avoid null when HTML section isn't present)
function tremorEl(id) { return document.getElementById(id); }

function setupTremorEventListeners() {
    const uploadArea = tremorEl('tremorUploadArea');
    const fileInput = tremorEl('tremorFileInput');
    const removeBtn = tremorEl('tremorRemoveFile');
    const analyzeBtn = tremorEl('tremorAnalyzeBtn');
    const loadDemoBtn = tremorEl('tremorLoadDemo');
    const anotherBtn = tremorEl('tremorAnalyzeAnother');
    const downloadBtn = tremorEl('tremorDownloadReport');

    if (!uploadArea) return;   // section not in page

    uploadArea.addEventListener('click', (e) => {
        if (e.target.tagName !== 'LABEL' && e.target.tagName !== 'INPUT')
            fileInput.click();
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) setTremorFile(e.target.files[0]);
    });
    uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        const f = e.dataTransfer.files[0];
        if (f && f.name.toLowerCase().endsWith('.csv')) setTremorFile(f);
    });

    if (removeBtn) removeBtn.addEventListener('click', (e) => { e.stopPropagation(); clearTremorFile(); });
    if (analyzeBtn) analyzeBtn.addEventListener('click', analyzeTremor);
    if (loadDemoBtn) loadDemoBtn.addEventListener('click', loadTremorDemo);
    if (anotherBtn) anotherBtn.addEventListener('click', resetTremorForm);
    if (downloadBtn) downloadBtn.addEventListener('click', downloadTremorReport);
}

function setTremorFile(file) {
    selectedTremorFile = file;
    tremorEl('tremorFileName').textContent = file.name;
    tremorEl('tremorSelectedFile').style.display = 'flex';
    tremorEl('tremorUploadArea').style.display = 'none';
    tremorEl('tremorAnalyzeBtn').disabled = false;
}

function clearTremorFile() {
    selectedTremorFile = null;
    const fi = tremorEl('tremorFileInput');
    if (fi) fi.value = '';
    tremorEl('tremorSelectedFile').style.display = 'none';
    tremorEl('tremorUploadArea').style.display = 'block';
    tremorEl('tremorAnalyzeBtn').disabled = true;
}

function resetTremorForm() {
    clearTremorFile();
    tremorEl('tremorResultsSection').style.display = 'none';
    tremorResults = null;
    window.scrollTo({ top: tremorEl('tremorUploadArea')?.offsetTop - 40 || 0, behavior: 'smooth' });
}

// ── Status ────────────────────────────────────────────────────────────────────
async function checkTremorStatus() {
    try {
        const resp = await fetch('/api/tremor-status');
        const data = await resp.json();
        const badge = tremorEl('tremorModelBadge');
        if (!badge) return;
        if (data.model_loaded) {
            badge.innerHTML = '<span class="badge badge-on">Model Loaded</span>';
        } else {
            badge.innerHTML = '<span class="badge badge-off">Model Not Loaded</span>';
        }
    } catch { /* ignore */ }
}

// ── Demo files ────────────────────────────────────────────────────────────────
async function loadTremorDemoFiles() {
    const sel = tremorEl('tremorDemoSelect');
    if (!sel) return;
    try {
        const resp = await fetch('/api/tremor-demo-files');
        const data = await resp.json();
        (data.files || []).forEach(f => {
            const opt = document.createElement('option');
            opt.value = f;
            opt.textContent = f;
            sel.appendChild(opt);
        });
    } catch { /* ignore */ }
}

async function loadTremorDemo() {
    const sel = tremorEl('tremorDemoSelect');
    const filename = sel?.value;
    if (!filename) return;

    // Fetch the actual CSV from the backend demo dir
    try {
        const resp = await fetch(`/api/tremor-demo-file/${filename}`);
        if (!resp.ok) throw new Error('Could not fetch demo file');
        const blob = await resp.blob();
        const file = new File([blob], filename, { type: 'text/csv' });
        setTremorFile(file);
    } catch (e) {
        alert('Error loading demo file: ' + e.message);
    }
}

// ── Analysis ──────────────────────────────────────────────────────────────────
async function analyzeTremor() {
    if (!selectedTremorFile) return;
    tremorEl('tremorLoadingOverlay').style.display = 'flex';

    const formData = new FormData();
    formData.append('file', selectedTremorFile);

    try {
        const resp = await fetch('/api/tremor-analyze', { method: 'POST', body: formData });
        if (!resp.ok) throw new Error('Server error');
        const data = await resp.json();
        if (data.success) {
            tremorResults = data;
            displayTremorResults(data);
        } else {
            alert(data.error || 'Analysis failed');
        }
    } catch (e) {
        alert('Error: ' + e.message);
    } finally {
        tremorEl('tremorLoadingOverlay').style.display = 'none';
    }
}

// ── Display results ───────────────────────────────────────────────────────────
function displayTremorResults(data) {
    const section = tremorEl('tremorResultsSection');
    section.style.display = 'block';
    section.scrollIntoView({ behavior: 'smooth' });

    // Verdict
    const box = tremorEl('tremorDecisionBox');
    const icon = tremorEl('tremorResultIcon');
    const title = tremorEl('tremorResultTitle');
    const desc = tremorEl('tremorResultDescription');

    if (data.tremor_detected) {
        box.className = 'result-box tremor-detected';
        icon.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
        title.textContent = 'TREMOR DETECTED';
        desc.textContent = `${data.tremor_window_pct}% of windows show tremor activity (confidence ${(data.confidence * 100).toFixed(1)}%)`;
    } else {
        box.className = 'result-box no-tremor';
        icon.innerHTML = '<i class="fas fa-check-circle"></i>';
        title.textContent = 'No Tremor Detected';
        desc.textContent = `Signal appears normal (confidence ${((1 - data.confidence) * 100).toFixed(1)}%)`;
    }

    // Metrics
    tremorEl('tremorDuration').textContent = data.duration_s + 's';
    tremorEl('tremorSamples').textContent = data.signal_length.toLocaleString();
    tremorEl('tremorWindows').textContent = data.num_windows;
    tremorEl('tremorConfidence').textContent = (data.confidence * 100).toFixed(1) + '%';
    tremorEl('tremorWindowPct').textContent = data.tremor_window_pct + '%';
    tremorEl('tremorProgressFill').style.width = (data.confidence * 100) + '%';

    // Color the progress bar
    const fill = tremorEl('tremorProgressFill');
    fill.style.background = data.tremor_detected
        ? 'linear-gradient(90deg, var(--warning-color), var(--danger-color))'
        : 'linear-gradient(90deg, var(--primary-color), var(--success-color))';

    // Timeline
    const container = tremorEl('tremorTimelineContainer');
    const timeline = tremorEl('tremorTimeline');
    const windows = data.window_results || [];
    if (windows.length > 0) {
        container.style.display = 'block';
        timeline.innerHTML = '';
        windows.forEach(w => {
            const bar = document.createElement('div');
            bar.className = 'tremor-bar ' + (w.probability >= 0.5 ? 'tremor-bar-pos' : 'tremor-bar-neg');
            bar.style.height = Math.max(4, w.probability * 100) + '%';
            bar.title = `${w.start_s}s – ${w.end_s}s  P=${(w.probability * 100).toFixed(1)}%  ${w.label}`;
            timeline.appendChild(bar);
        });
    } else {
        container.style.display = 'none';
    }
}

// ── Report download ───────────────────────────────────────────────────────────
function downloadTremorReport() {
    if (!tremorResults) return;
    const d = tremorResults;
    const lines = [
        '='.repeat(70),
        '  TREMOR DETECTION REPORT',
        '='.repeat(70),
        '',
        'Generated : ' + new Date().toLocaleString(),
        'File      : ' + (d.filename || 'N/A'),
        '',
        '--- VERDICT ---',
        'Tremor Detected : ' + (d.tremor_detected ? 'YES' : 'NO'),
        'Confidence      : ' + (d.confidence * 100).toFixed(1) + '%',
        'Tremor windows  : ' + d.tremor_window_pct + '%',
        '',
        '--- SIGNAL ---',
        'Duration  : ' + d.duration_s + 's',
        'Samples   : ' + d.signal_length,
        'Windows   : ' + d.num_windows,
        '',
        '--- PER-WINDOW ---',
    ];
    (d.window_results || []).forEach(w => {
        lines.push(`  [${w.window_idx}] ${w.start_s}s-${w.end_s}s  P=${(w.probability * 100).toFixed(1)}%  ${w.label}`);
    });
    lines.push('', '='.repeat(70), '  END OF REPORT', '='.repeat(70));

    const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `tremor_report_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
}