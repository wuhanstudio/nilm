// Upload modal logic
let activeBuildingId = null, selectedFile = null;

function openModal(buildingId, buildingLabel) {
    activeBuildingId = buildingId; selectedFile = null;
    document.getElementById('modal-building-name').textContent = buildingLabel;
    document.getElementById('csv-file-input').value = '';
    document.getElementById('file-pill-wrap').style.display = 'none';
    document.getElementById('upload-progress').style.display = 'none';
    document.getElementById('upload-feedback').textContent = '';
    document.getElementById('confirm-upload-btn').disabled = true;
    setProgressBar(0);
    $('#uploadModal').modal('show');
}

document.getElementById('csv-file-input').addEventListener('change', function() {
    if (this.files && this.files[0]) handleFileSelected(this.files[0]);
});

const dropZone = document.getElementById('drop-zone');
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0]; if (file) handleFileSelected(file);
});

function handleFileSelected(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) { showFeedback('Only .csv files are accepted.', 'danger'); return; }
    selectedFile = file;
    document.getElementById('file-name-display').textContent = file.name;
    document.getElementById('file-pill-wrap').style.display = 'block';
    document.getElementById('confirm-upload-btn').disabled = false;
    document.getElementById('upload-feedback').textContent = '';
    document.getElementById('upload-progress').style.display = 'none';
    setProgressBar(0);
}

document.getElementById('remove-file-btn').addEventListener('click', () => {
    selectedFile = null;
    document.getElementById('csv-file-input').value = '';
    document.getElementById('file-pill-wrap').style.display = 'none';
    document.getElementById('confirm-upload-btn').disabled = true;
    document.getElementById('upload-feedback').textContent = '';
});

document.getElementById('confirm-upload-btn').addEventListener('click', () => {
    if (!selectedFile || !activeBuildingId) return;
    console.log("Selected file for upload:", selectedFile);
    console.log("Active building ID:", activeBuildingId);
    Papa.parse(selectedFile, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: async function(results) {
            const redd_submission = {};

            results.meta.fields.forEach(field => {
                redd_submission[field] = results.data.map(row => row[field] ?? 0);
            });

            console.log("Submission:", redd_submission);

            const redd_gt = await loadBuildingCSV(`redd/building_${activeBuildingId}_combined.csv`);
            console.log("Ground Truth:", redd_gt);

            // Match if uploaded file has same structure as ground truth
            const hasAllFields = Object.keys(redd_gt).every(key => key in redd_submission);
            if (!hasAllFields) {
                showFeedback('Uploaded CSV structure does not match expected format.', 'danger');
                return;
            }

            // Match if labels are identical
            const labelsMatch = JSON.stringify(redd_submission.labels) === JSON.stringify(redd_gt.labels);
            if (!labelsMatch) {
                showFeedback('Timestamps (labels) in uploaded CSV do not match expected format.', 'danger');
                return;
            }

            // Match array lengths for each appliance
            const appliances = Object.keys(redd_gt).filter(k => k !== 'labels');
            const lengthMismatch = appliances.some(app => redd_submission[app].length !== redd_gt[app].length);
            if (lengthMismatch) {
                showFeedback('One or more appliance columns in uploaded CSV do not have the expected number of entries.', 'danger');
                return;
            }

             onUploadSuccess(activeBuildingId, selectedFile, redd_submission, redd_gt);
            // initializeChart(redd_submission);
        }
    });
});

function onUploadSuccess(buildingId, file, submission, groundTruth) {
    showFeedback('Upload successful!', 'success');
    rowState[buildingId].uploadedFile = file.name;
    const statusCell = document.getElementById(`${buildingId}_status`);
    statusCell.innerHTML = '<span class="status-badge success">✓ Uploaded</span>';
    setTimeout(() => $('#uploadModal').modal('hide'), 1200);

    // For each appiance, match submission with ground truth and calculate a dummy score (e.g., mean absolute error)
    submission.appliances = Object.keys(submission).filter(k => k !== 'main');
    console.log("Appliances in submission:", submission.appliances);

    submission.appliances.forEach(app => {
        const subValues = submission[app];
        console.log(subValues);

        // Calculate accuracy as percentage of correctly identified on/off states
        let correct = 0;
        for (let i = 0; i < subValues.length; i++) {
            // Using a noisy threshold
            threshold = rowState[buildingId].threshold;
            const gt = groundTruth[app][i] > threshold ? 1 : 0;
            if (subValues[i] === gt) correct++;
        }
        const accuracy = (correct / subValues.length) * 100;
        console.log(`Accuracy for ${app}: ${accuracy.toFixed(2)}%`);

        // Update the "Your Score" cell with this dummy accuracy
        if (app.toLowerCase() === 'fridge') document.getElementById(`${buildingId}_${app.toLowerCase()}`).textContent = accuracy.toFixed(2) + '%';
        if (app.toLowerCase() === 'microwave') document.getElementById(`${buildingId}_${app.toLowerCase()}`).textContent = accuracy.toFixed(2) + '%';
    });
}

function setProgressBar(pct) {
    const bar = document.getElementById('progress-bar-fill');
    bar.style.width = pct + '%'; bar.textContent = pct + '%';
}

function showFeedback(msg, type) {
    const colors = { info: '#004085', success: '#155724', danger: '#721c24' };
    const el = document.getElementById('upload-feedback');
    el.textContent = msg; el.style.color = colors[type] || '#333';
}
