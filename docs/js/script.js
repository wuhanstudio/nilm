// ── Table rows ────────────────────────────────────────────────────────────
const buildings = [
    { id: '1', label: 'Building 1' },
    { id: '2', label: 'Building 2' },
    { id: '3', label: 'Building 3' },
    { id: '4', label: 'Building 4' },
    { id: '5', label: 'Building 5' },
    { id: '6', label: 'Building 6' },
];
const rowState = {};
buildings.forEach(b => { rowState[b.id] = { fridgeScore: '-', microwaveScore: '-', threshold: 80, uploadedFile: null }; });

function renderRows() {
    const tbody = document.querySelector('tbody');
    tbody.innerHTML = '';
    buildings.forEach(b => {
        const s = rowState[b.id];
        const tr = document.createElement('tr');
        tr.id = `row-${b.id}`;
        tr.innerHTML = `
        <th scope="row">${b.label}</th>
        <td id="${b.id}_fridge">${s.fridgeScore}</td>
        <td id="${b.id}_microwave">${s.microwaveScore}</td>
        <td id="${b.id}_threshold"><input type="number" class="form-control" value="${s.threshold}" id="${b.id}_threshold-input" /></td>
        <td><div id="${b.id}_download"><a href="redd/building_${b.id}_combined.csv" target="_blank"><span class="badge badge-success" style="cursor:pointer;">csv</span></a></div></td>
        <td><button class="btn-upload open-upload-modal" data-building-id="${b.id}" data-building-label="${b.label}"><i class="fas fa-upload mr-1"></i>Upload</button></td>
        <td id="${b.id}_status"><span class="status-badge pending" style="display:${s.uploadedFile ? 'inline-block' : 'none'}">${s.uploadedFile ? 'Uploaded' : ''}</span></td>
        `;
        tbody.appendChild(tr);
    });
    document.querySelectorAll('.open-upload-modal').forEach(btn => {
        btn.addEventListener('click', () => openModal(btn.dataset.buildingId, btn.dataset.buildingLabel));
    });
}

// ── Appliance chart data ──────────────────────────────────────────────────

async function loadBuildingCSV(csvUrl) {
    const response = await fetch(csvUrl);
    const csvText = await response.text();

    const lines = csvText.trim().split("\n");
    const headers = lines[0].split(",");

    const RAW = {};

    headers.forEach(h => RAW[h.trim()] = []);

    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(",");

        values.forEach((value, index) => {
            RAW[headers[index].trim()].push(Number(value) || 0);
        });
    }

    return RAW;
}

function initializeChart(RAW, buildingId) {
    // Generate colors dynamically for appliances
    const COLORS = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db','#9b59b6','#1abc9c','#16a085','#8e44ad','#d35400'];

    function hexAlpha(hex, a) {
        const r = parseInt(hex.slice(1,3),16);
        const g = parseInt(hex.slice(3,5),16);
        const b = parseInt(hex.slice(5,7),16);
        return `rgba(${r},${g},${b},${a})`;
    }

    const applianceKeys = Object.keys(RAW);  // automatically detected appliances
    // Remove 'labels' if it exists, since it's not an appliance
    const labelIdx = applianceKeys.indexOf('labels');
    if (labelIdx > -1) applianceKeys.splice(labelIdx, 1);

    const datasets = applianceKeys.map((key,i) => ({
        label: key.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase()),  // nice label
        data: RAW[key],
        borderColor: COLORS[i % COLORS.length],
        backgroundColor: hexAlpha(COLORS[i % COLORS.length], 0.08),
        borderWidth: 1.2,
        pointRadius: 0,
        tension: 0.2,
        fill: false,
        hidden: false,
    }));

    const ctx = document.getElementById(`building-${buildingId}-chart`).getContext('2d');
    const chart = new Chart(ctx, {
        type: 'line',
        data: { labels: RAW.labels, datasets },
        options: {
            animation: false,
            responsive: true,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)} W`
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    ticks: {
                        maxTicksLimit: 12,
                        callback: (val, i) => `#${RAW.labels[i]}`,
                        font: { family: 'Ubuntu Mono', size: 10 }
                    },
                    grid: { color: '#f0f0f0' }
                },
                y: {
                    display: true,
                    title: { display: true, text: 'Power (W)', font: { family: 'Ubuntu Mono', size: 11 } },
                    ticks: { font: { family: 'Ubuntu Mono', size: 10 } },
                    grid: { color: '#f0f0f0' }
                }
            }
        }
    });

    // Toggle buttons
    const togglesEl = document.getElementById(`appliance-toggles-${buildingId}`);
    const pillsEl   = document.getElementById(`stat-pills-${buildingId}`);

    applianceKeys.forEach((key,i) => {
        const ds = chart.data.datasets[i];
        const color = COLORS[i % COLORS.length];

        const btn = document.createElement('button');
        btn.className = 'toggle-btn active';
        btn.style.borderColor = color;
        btn.style.backgroundColor = color;
        btn.style.color = '#fff';
        btn.innerHTML = `<span class="dot" style="background:${color}"></span>${ds.label}`;
        btn.dataset.idx = i;
        btn.addEventListener('click', function() {
            const idx = +this.dataset.idx;
            const hidden = chart.isDatasetVisible(idx);
            if(hidden){
                chart.hide(idx);
                this.classList.remove('active');
                this.style.backgroundColor = '';
                this.style.color = color;
            } else {
                chart.show(idx);
                this.classList.add('active');
                this.style.backgroundColor = color;
                this.style.color = '#fff';
            }
        });
        togglesEl.appendChild(btn);

        // Optional: show basic stats dynamically
        const dataArr = RAW[key];
        const mean = (dataArr.reduce((a,b)=>a+b,0)/dataArr.length).toFixed(2);
        const max = Math.max(...dataArr);
        const pill = document.createElement('div');
        pill.className = 'stat-pill';
        pill.innerHTML = `<strong style="color:${color}">${ds.label}</strong> avg ${mean}W &bull; max ${max}W`;
        pillsEl.appendChild(pill);
    });
}

// Initialize chart with first building's data by default
async function renderChart() {
    const section = document.getElementById('chart-section');

    for(let i=1; i<=6; i++){
        // Create building container
        const buildingDiv = document.createElement('div');
        buildingDiv.className = 'container';
        buildingDiv.id = `building-${i}`;
        buildingDiv.innerHTML = `
            <h3>Building ${i} — Appliance Power Readings</h3>
            <p class="subtitle">Downsampled readings &bull; Power in Watts &bull; Toggle appliances below</p>
            <div class="appliance-toggles-${i}" id="appliance-toggles-${i}"></div>
            <div class="chart-wrapper">
                <canvas id="building-${i}-chart"></canvas>
            </div>
            <div class="stat-pills" id="stat-pills-${i}"></div>
            <br />
            <hr />
        `;
        section.appendChild(buildingDiv);

        // Load CSV dynamically
        const RAW = await loadBuildingCSV(`redd/building_${i}_combined.csv`); // Replace with your CSV paths

        // Only use first 1500 entries for faster loading in this demo
        Object.keys(RAW).forEach(k => {
            RAW[k] = RAW[k].slice(0, 1500);

            // Index the labels
            RAW.labels = RAW.labels || [];
        });
        RAW.labels = RAW["main"].map((_, i) => i + 1);
        console.log("Loaded JSON:", RAW);

        initializeChart(RAW, i);
    }
}

// Initialize table rows
renderRows();

// Initialize charts
renderChart();

$( document ).ready( () => {
    $('body').scrollspy({ target: '#main-nav', offset: 130 })

    // Navbar click scroll
    $(".navbar a").on('click', function(event) {
        // Make sure this.hash has a value before overriding default behavior
        if (this.hash !== "") {
            // Prevent default anchor click behavior
            event.preventDefault();

            // Store hash
            var hash = this.hash;

            // Using jQuery's animate() method to add smooth page scroll
            // The optional number (800) specifies the number of milliseconds it takes to scroll to the specified area
            var offset = -100;
            $('html, body').animate({
                scrollTop: ($(hash).offset().top + offset)
            }, 1000, function(){
                // Add hash (#) to URL when done scrolling (default click behavior)
                // window.location.hash = hash;
            });
        }
    });

    // Update threshold value in rowState when user changes it in the table
    document.querySelectorAll('input[type="number"]').forEach(input => {
        input.addEventListener('change', function() {
            console.log("Threshold input changed:", this.value, "for input ID:", this.id);
            const buildingId = this.id.split('_')[0];
            rowState[buildingId].threshold = +this.value;
            console.log(`Updated threshold for Building ${buildingId}:`, rowState[buildingId].threshold);
        });
    });

});
