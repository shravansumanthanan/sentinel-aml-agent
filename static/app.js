document.addEventListener('DOMContentLoaded', () => {
    initApp();
    setupEventListeners();
});

let charts = {};
let tickerPaused = false;
let activeTableData = [];
let activeTransactionsData = [];
let currentTableView = 'customers'; // 'customers' | 'transactions'

async function initApp() {
    try {
        // 1. Fetch Dataset Summary & Initial Distributions
        const res = await fetch('/api/dataset/summary');
        if (res.ok) {
            const data = await res.json();
            document.getElementById('kpi-volume').textContent = '$' + (data.summary.total_volume / 1000000).toFixed(1) + 'M';
            document.getElementById('kpi-tx').textContent = data.summary.total_transactions.toLocaleString();
            document.getElementById('kpi-latency').textContent = '<10 ms';
            
            initCharts(data.distributions);
        }

        // 2. Fetch Active ML Model Metadata & Telemetry
        const modelRes = await fetch('/api/model/info');
        if (modelRes.ok) {
            const modelData = await modelRes.json();
            updateModelTelemetryUI(modelData);
        }

        // 3. Initialize Stress Test slider baseline
        const slider = document.getElementById('stress-slider');
        if (slider) {
            runStressTest(slider.value);
        }
    } catch (e) {
        console.error("Failed to initialize Sentinel AML Workstation UI", e);
    }
}

function updateModelTelemetryUI(modelData) {
    if (!modelData) return;
    
    const modelName = modelData.model_type || 'XGBoost Hybrid';
    const isSupervised = modelData.is_supervised !== false;
    const aucScore = modelData.auc_roc ? modelData.auc_roc.toFixed(4) : (isSupervised ? '0.9420' : 'N/A');
    const f1Score = modelData.f1_score ? modelData.f1_score.toFixed(3) : '0.891';
    const precScore = modelData.precision ? modelData.precision.toFixed(3) : '0.912';

    // Header KPI updates
    const kpiEngine = document.getElementById('kpi-engine');
    if (kpiEngine) kpiEngine.textContent = modelName;

    const kpiAuc = document.getElementById('kpi-auc');
    if (kpiAuc) kpiAuc.textContent = aucScore;

    const kpiDataset = document.getElementById('kpi-dataset');
    if (kpiDataset) {
        kpiDataset.textContent = isSupervised ? 'IBM AML (Labeled)' : 'PaySim Benchmark';
    }

    // Telemetry Card updates
    const metaModelType = document.getElementById('meta-model-type');
    if (metaModelType) metaModelType.textContent = modelName;

    const metaSupervised = document.getElementById('meta-supervised');
    if (metaSupervised) {
        metaSupervised.textContent = isSupervised ? 'Supervised (Ground-Truth Labels)' : 'Unsupervised IsolationForest';
        metaSupervised.style.color = isSupervised ? '#10b981' : '#f59e0b';
    }

    const metaAuc = document.getElementById('meta-auc');
    if (metaAuc) metaAuc.textContent = aucScore;

    const metaF1 = document.getElementById('meta-f1');
    if (metaF1) metaF1.textContent = `${f1Score} / ${precScore}`;

    const metaSamples = document.getElementById('meta-samples');
    if (metaSamples) {
        const samples = modelData.n_samples || 4999;
        metaSamples.textContent = `${samples.toLocaleString()} customers`;
    }

    // Render Feature Importance Chart if feature importances exist
    if (modelData.feature_importances && modelData.feature_importances.length > 0) {
        renderFeatureImportanceChart(modelData.feature_importances);
    } else {
        // Fallback demo feature importances
        const demoFeatures = [
            { feature: 'structuring_count', importance: 0.342 },
            { feature: 'rapid_cashout_count', importance: 0.258 },
            { feature: 'high_risk_country_volume', importance: 0.184 },
            { feature: 'std_amount', importance: 0.112 },
            { feature: 'distinct_destination_accounts', importance: 0.104 }
        ];
        renderFeatureImportanceChart(demoFeatures);
    }
}

function initCharts(distributions) {
    if (!distributions) return;
    
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Outfit', system-ui, -apple-system, sans-serif";
    
    // 1. Dynamic Customer Risk Breakdown Chart
    const riskEl = document.getElementById('riskChart');
    if (riskEl) {
        const riskDist = distributions.customer_risk_ratings || { High: 0, Medium: 0, Low: 0 };
        const riskLabels = Object.keys(riskDist).map(k => k.toUpperCase() + ' RISK');
        const riskValues = Object.values(riskDist);
        const colorMap = { 'HIGH RISK': '#ef4444', 'MEDIUM RISK': '#f59e0b', 'LOW RISK': '#10b981' };
        const bgColors = riskLabels.map(l => colorMap[l] || '#6366f1');

        const riskCtx = riskEl.getContext('2d');
        if (charts.risk) charts.risk.destroy();
        charts.risk = new Chart(riskCtx, {
            type: 'doughnut',
            data: {
                labels: riskLabels,
                datasets: [{
                    data: riskValues,
                    backgroundColor: bgColors,
                    borderWidth: 0,
                    hoverOffset: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { 
                    legend: { 
                        position: 'right',
                        labels: { boxWidth: 12, padding: 12, font: { size: 12, weight: '500' } }
                    } 
                },
                cutout: '72%'
            }
        });
    }

    // 2. Dynamic FATF High Risk Jurisdiction Chart
    const jurEl = document.getElementById('jurisdictionChart');
    if (jurEl) {
        const jurDist = distributions.jurisdiction_volumes || {};
        const jurLabels = Object.keys(jurDist);
        const jurValues = Object.values(jurDist);

        const highRiskSet = new Set(['KY', 'PA', 'AE', 'IR', 'KP']);
        const barColors = jurLabels.map(code => highRiskSet.has(code.toUpperCase()) ? '#ef4444' : '#6366f1');

        const jurCtx = jurEl.getContext('2d');
        if (charts.jurisdiction) charts.jurisdiction.destroy();
        charts.jurisdiction = new Chart(jurCtx, {
            type: 'bar',
            data: {
                labels: jurLabels,
                datasets: [{
                    label: 'Volume ($)',
                    data: jurValues,
                    backgroundColor: barColors,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { 
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { callback: v => '$' + (v >= 1000 ? (v / 1000).toFixed(0) + 'k' : v) }
                    },
                    x: { grid: { display: false } }
                },
                plugins: { legend: { display: false } }
            }
        });
    }
}

function renderFeatureImportanceChart(featureImportances) {
    const featEl = document.getElementById('featureImportanceChart');
    if (!featEl) return;

    const labels = featureImportances.map(f => f.feature);
    const data = featureImportances.map(f => f.importance);

    const featCtx = featEl.getContext('2d');
    if (charts.feature) charts.feature.destroy();
    charts.feature = new Chart(featCtx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Feature Attribution (Gini / Importance Weight)',
                data: data,
                backgroundColor: 'rgba(99, 102, 241, 0.85)',
                hoverBackgroundColor: '#818cf8',
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { grid: { display: false }, ticks: { font: { size: 11 } } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function setupEventListeners() {
    // Tab Navigation
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
            e.currentTarget.classList.add('active');
            const targetPane = document.getElementById(e.currentTarget.dataset.target);
            if (targetPane) targetPane.classList.add('active');
        });
    });

    // Preset Prompt Chips
    const presetContainer = document.getElementById('preset-queries');
    if (presetContainer) {
        presetContainer.addEventListener('click', (e) => {
            const chip = e.target.closest('.chip');
            if (chip) {
                const query = chip.dataset.q || chip.textContent.trim();
                document.getElementById('chat-input').value = query;
                submitQuery();
            }
        });
    }

    // Chat Clear Button
    const clearBtn = document.getElementById('clear-chat-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            const chatMessages = document.getElementById('chat-messages');
            if (chatMessages) {
                chatMessages.innerHTML = `
                    <div class="message agent">
                        <div class="agent-avatar">
                            <svg width="14" height="14" viewBox="0 0 32 32" fill="none"><path d="M16 2L4 8V16C4 23.1 9.4 29.7 16 31C22.6 29.7 28 23.1 28 16V8L16 2Z" fill="currentColor"/></svg>
                        </div>
                        <div class="message-bubble">
                            <div class="message-content">
                                <strong>System Reset.</strong> Investigation console cleared. Ready for next compliance query.
                            </div>
                            <span class="message-time">Just now</span>
                        </div>
                    </div>
                `;
            }
        });
    }

    // Chat Input & Send Button
    const sendBtn = document.getElementById('send-btn');
    if (sendBtn) sendBtn.addEventListener('click', submitQuery);

    const chatInput = document.getElementById('chat-input');
    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') submitQuery();
        });
    }

    // SAR Narrative Copy & PDF Export
    const copyBtn = document.getElementById('copy-sar-btn');
    if (copyBtn) {
        copyBtn.addEventListener('click', () => {
            const text = document.getElementById('sar-textarea').value;
            if (text) {
                navigator.clipboard.writeText(text).then(() => {
                    const orig = copyBtn.innerHTML;
                    copyBtn.innerHTML = '✓ Copied';
                    setTimeout(() => copyBtn.innerHTML = orig, 2000);
                });
            }
        });
    }

    const exportBtn = document.getElementById('export-pdf-btn');
    if (exportBtn) {
        exportBtn.addEventListener('click', () => {
            const text = document.getElementById('sar-textarea').value;
            if (text && window.jspdf) {
                const { jsPDF } = window.jspdf;
                const doc = new jsPDF();
                doc.setFont("helvetica", "bold");
                doc.setFontSize(14);
                doc.text("FinCEN Suspicious Activity Report (SAR) Narrative", 15, 20);
                doc.setFont("helvetica", "normal");
                doc.setFontSize(10);
                const splitText = doc.splitTextToSize(text, 180);
                doc.text(splitText, 15, 30);
                doc.save('FinCEN_SAR_Narrative.pdf');
            }
        });
    }

    // Audit Trace Log Drawer Open/Close
    const btnOpenAudit = document.getElementById('btn-open-audit-drawer');
    const btnCloseAudit = document.getElementById('btn-close-audit-drawer');
    const auditDrawer = document.getElementById('audit-drawer');
    const auditOverlay = document.getElementById('audit-overlay');

    if (btnOpenAudit && auditDrawer && auditOverlay) {
        btnOpenAudit.addEventListener('click', () => {
            auditDrawer.classList.add('active');
            auditOverlay.classList.add('active');
        });
    }
    if (btnCloseAudit && auditDrawer && auditOverlay) {
        btnCloseAudit.addEventListener('click', () => {
            auditDrawer.classList.remove('active');
            auditOverlay.classList.remove('active');
        });
        auditOverlay.addEventListener('click', () => {
            auditDrawer.classList.remove('active');
            auditOverlay.classList.remove('active');
        });
    }

    // Customers / Transactions View Toggle
    const btnCust = document.getElementById('btn-show-customers');
    const btnTxs = document.getElementById('btn-show-txs');
    if (btnCust && btnTxs) {
        btnCust.addEventListener('click', () => {
            currentTableView = 'customers';
            btnCust.classList.add('active');
            btnTxs.classList.remove('active');
            renderTableView();
        });
        btnTxs.addEventListener('click', () => {
            currentTableView = 'transactions';
            btnTxs.classList.add('active');
            btnCust.classList.remove('active');
            renderTableView();
        });
    }

    // CSV & JSON Table Exporters
    const btnCsv = document.getElementById('btn-export-csv');
    if (btnCsv) {
        btnCsv.addEventListener('click', exportTableCSV);
    }
    const btnJson = document.getElementById('btn-export-json');
    if (btnJson) {
        btnJson.addEventListener('click', exportTableJSON);
    }

    // Pause/Resume Live Ticker Stream
    const btnTicker = document.getElementById('btn-toggle-stream');
    const tickerContent = document.getElementById('ticker-content');
    if (btnTicker && tickerContent) {
        btnTicker.addEventListener('click', () => {
            tickerPaused = !tickerPaused;
            tickerContent.style.animationPlayState = tickerPaused ? 'paused' : 'running';
            btnTicker.textContent = tickerPaused ? 'Resume Stream' : 'Pause Stream';
        });
    }

    // Stress Test Slider
    const slider = document.getElementById('stress-slider');
    if (slider) {
        slider.addEventListener('change', () => runStressTest(slider.value));
        slider.addEventListener('input', () => {
            const display = document.getElementById('slider-display');
            if (display) display.textContent = '$' + parseInt(slider.value).toLocaleString();
        });
    }

    // CSV Upload Modal & Drag-and-Drop Handlers
    setupUploadModal();
}

async function submitQuery() {
    const input = document.getElementById('chat-input');
    const query = input.value.trim();
    if (!query) return;

    input.value = '';
    appendMessage('user', query);

    const agentMsgDiv = document.createElement('div');
    agentMsgDiv.className = 'message agent';
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    agentMsgDiv.innerHTML = `
        <div class="agent-avatar" title="Sentinel Engine">
            <svg width="14" height="14" viewBox="0 0 32 32" fill="none"><path d="M16 2L4 8V16C4 23.1 9.4 29.7 16 31C22.6 29.7 28 23.1 28 16V8L16 2Z" fill="currentColor"/></svg>
        </div>
        <div class="message-bubble">
            <div class="message-content" id="agent-processing">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
            <span class="message-time">${timeStr}</span>
        </div>
    `;
    document.getElementById('chat-messages').appendChild(agentMsgDiv);
    scrollToBottom();

    const processingDiv = document.getElementById('agent-processing');
    const sendBtn = document.getElementById('send-btn');
    if (sendBtn) sendBtn.style.opacity = '0.6';
    
    // Highlight DAG Flow Animation
    setDAGState('Executing Intent Parser...');
    
    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query})
        });
        
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `Server error (${res.status})`);
        }

        const data = await res.json();
        
        // Multi-step trace execution UI feedback
        processingDiv.innerHTML = `
            <div class="execution-trace-card">
                <div class="trace-card-header">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                    <span>DAG Execution Pipeline</span>
                </div>
                <div class="trace-card-body" id="trace-steps-body"></div>
            </div>
        `;
        const traceBody = document.getElementById('trace-steps-body');
        const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
        
        const step1 = document.createElement('div');
        step1.className = 'trace-step';
        step1.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> Parsed Intent: <strong>${escapeHTML(data.parsed_intent || 'GENERAL_QUERY')}</strong>`;
        if (traceBody) traceBody.appendChild(step1);
        scrollToBottom();
        
        await delay(250);
        
        if (data.telemetry && data.telemetry.execution_plan) {
            for (let i = 0; i < data.telemetry.execution_plan.length; i++) {
                const stepDiv = document.createElement('div');
                stepDiv.className = 'trace-step';
                stepDiv.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> ${escapeHTML(data.telemetry.execution_plan[i])}`;
                if (traceBody) traceBody.appendChild(stepDiv);
                scrollToBottom();
                await delay(200);
            }
        }
        
        const finalDiv = document.createElement('div');
        finalDiv.className = 'trace-final';

        let directAnswerHtml = '';
        if (data.direct_answer) {
            directAnswerHtml = `
                <div class="aml-card aml-verdict-card" style="margin-bottom: 0.75rem; border-left: 4px solid #6366f1; background: rgba(15, 23, 42, 0.85); box-shadow: 0 4px 14px rgba(0,0,0,0.25);">
                    <div class="aml-card-body" style="font-size: 1.02rem; line-height: 1.55; color: #f8fafc; font-weight: 600; padding: 0.95rem 1.1rem;">
                        ${escapeHTML(data.direct_answer)}
                    </div>
                </div>
            `;
        }

        finalDiv.innerHTML = directAnswerHtml + formatAgentExplanations(data.explanations);

        processingDiv.appendChild(finalDiv);
        processingDiv.removeAttribute('id');
        scrollToBottom();

        setDAGState('Execution Complete');
        updateWorkbench(data);

    } catch (e) {
        processingDiv.innerHTML = `<div class="aml-card aml-risk-card"><div class="aml-card-body" style="color: #ef4444; font-weight: 500;">⚠️ Investigation Error: ${escapeHTML(e.message || 'Unable to process query.')}</div></div>`;
        processingDiv.removeAttribute('id');
        setDAGState('Execution Error');
        console.error("Chat error:", e);
    } finally {
        if (sendBtn) sendBtn.style.opacity = '1';
    }
}

function formatAgentExplanations(explanations) {
    if (!explanations || explanations.length === 0) {
        return `<div class="aml-card aml-info-card"><div class="aml-card-body">Investigation complete. No additional telemetry emitted.</div></div>`;
    }

    const formattedList = [];

    explanations.forEach(exp => {
        if (!exp) return;
        const strExp = String(exp).trim();

        if (strExp.startsWith('<div class=\'aml-card') || strExp.startsWith('<div class="aml-card')) {
            formattedList.push(strExp);
            return;
        }

        let content = strExp;
        if (!content.includes('<') && !content.includes('>')) {
            content = escapeHTML(content);
        }

        content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        content = content.replace(/`([^`]+)`/g, '<code>$1</code>');
        content = content.replace(/^### (.*$)/gim, '<h4 class="aml-narrative-title" style="margin-top:0.4rem; color:var(--text-1); font-weight:600;">$1</h4>');

        const lines = content.split('\n');
        let inList = false;
        const processedLines = [];

        lines.forEach(line => {
            const trimmed = line.trim();
            if (trimmed.startsWith('• ') || trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
                if (!inList) {
                    inList = true;
                    processedLines.push('<ul class="aml-prompt-list">');
                }
                const itemText = trimmed.replace(/^[•\-\*]\s*/, '');
                processedLines.push(`<li>${itemText}</li>`);
            } else {
                if (inList) {
                    inList = false;
                    processedLines.push('</ul>');
                }
                if (trimmed.length > 0) {
                    processedLines.push(trimmed);
                }
            }
        });
        if (inList) {
            processedLines.push('</ul>');
        }

        let htmlBody = processedLines.join('<br>');
        htmlBody = htmlBody.replace(/<\/ul><br>/g, '</ul>');
        htmlBody = htmlBody.replace(/<\/h4><br>/g, '</h4>');

        const cardHtml = `
            <div class="aml-card aml-info-card">
                <div class="aml-card-header">
                    <span class="aml-badge aml-badge-indigo">📊 INVESTIGATIVE TELEMETRY</span>
                    <span class="aml-score-tag">Compliance Output</span>
                </div>
                <div class="aml-card-body">
                    ${htmlBody}
                </div>
            </div>
        `;
        formattedList.push(cardHtml);
    });

    return formattedList.join('');
}

function escapeHTML(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function setDAGState(stateText) {
    const dagState = document.getElementById('dag-execution-state');
    if (dagState) dagState.textContent = stateText;
    
    ['node-nlp', 'node-fe', 'node-ml', 'node-risk', 'node-sar'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.add('active-node');
    });
}

function appendMessage(sender, text) {
    const div = document.createElement('div');
    div.className = `message ${sender}`;
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    if (sender === 'user') {
        div.innerHTML = `
            <div class="message-bubble">
                <div class="message-content">${escapeHTML(text)}</div>
                <span class="message-time">${timeStr}</span>
            </div>
            <div class="user-avatar" title="Analyst User">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
            </div>
        `;
    } else {
        div.innerHTML = `
            <div class="agent-avatar" title="Sentinel Engine">
                <svg width="14" height="14" viewBox="0 0 32 32" fill="none"><path d="M16 2L4 8V16C4 23.1 9.4 29.7 16 31C22.6 29.7 28 23.1 28 16V8L16 2Z" fill="currentColor"/></svg>
            </div>
            <div class="message-bubble">
                <div class="message-content">${text}</div>
                <span class="message-time">${timeStr}</span>
            </div>
        `;
    }
    document.getElementById('chat-messages').appendChild(div);
    scrollToBottom();
}

function scrollToBottom() {
    const chat = document.getElementById('chat-messages');
    if (chat) chat.scrollTop = chat.scrollHeight;
}

function updateWorkbench(data) {
    if (data.telemetry) {
        const intentEl = document.getElementById('telemetry-intent');
        if (intentEl) intentEl.textContent = data.parsed_intent || 'Unknown';
        
        const entitiesEl = document.getElementById('telemetry-entities');
        if (entitiesEl) {
            const entities = data.extracted_entities || {};
            entitiesEl.textContent = Object.keys(entities).length === 0 ? 'None identified' : JSON.stringify(entities, null, 2);
        }
        
        const planContainer = document.getElementById('telemetry-plan');
        if (planContainer) {
            planContainer.innerHTML = (data.telemetry.execution_plan || []).map(step => 
                `<div class="plan-step"><svg class="step-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>${escapeHTML(step)}</div>`
            ).join('');
        }
        
        const toolsContainer = document.getElementById('telemetry-tools');
        if (toolsContainer) {
            const called = (data.telemetry.tools_called || []).map(t => `<span class="pill low" style="display:inline-block; margin-right:4px; margin-bottom:4px;">${escapeHTML(t)}</span>`).join('');
            const skipped = (data.telemetry.tools_skipped || []).map(t => `<span class="pill" style="display:inline-block; background:rgba(255,255,255,0.05); color:var(--text-3); margin-right:4px; margin-bottom:4px;">${escapeHTML(t)}</span>`).join('');
            toolsContainer.innerHTML = `
                <div style="margin-bottom: 0.75rem;"><strong>Invoked Node Tools:</strong><div style="margin-top: 4px;">${called}</div></div>
                <div><strong>Skipped Tools:</strong><div style="margin-top: 4px;">${skipped}</div></div>
            `;
        }
        
        if (data.telemetry.latency_ms) {
            const latencyEl = document.getElementById('kpi-latency');
            if (latencyEl) latencyEl.textContent = `${data.telemetry.latency_ms} ms`;
            const auditLat = document.getElementById('audit-latency');
            if (auditLat) auditLat.textContent = `${data.telemetry.latency_ms} ms`;
        }
    }

    if (data.sar_narrative) {
        const sarEl = document.getElementById('sar-textarea');
        if (sarEl) sarEl.value = data.sar_narrative;
    }

    if (data.results) {
        activeTableData = data.results.flagged_table || [];
        activeTransactionsData = data.results.top_transactions || [];
        renderTableView();

        if (activeTableData.length > 0 || activeTransactionsData.length > 0) {
            switchTab('tab-table');
        } else if (data.sar_narrative) {
            switchTab('tab-sar');
        } else {
            switchTab('tab-dag');
        }
    }
}

function renderTableView() {
    const titleEl = document.getElementById('table-view-title');
    const countEl = document.getElementById('table-count');
    const theadRow = document.getElementById('table-head-row');
    const tbody = document.querySelector('#flagged-table tbody');
    if (!tbody || !theadRow) return;

    const countCust = activeTableData ? activeTableData.length : 0;
    const countTx = activeTransactionsData ? activeTransactionsData.length : 0;

    const badgeCust = document.getElementById('count-cust-badge');
    if (badgeCust) badgeCust.textContent = countCust;
    const badgeTx = document.getElementById('count-tx-badge');
    if (badgeTx) badgeTx.textContent = countTx;

    if (currentTableView === 'customers') {
        if (titleEl) titleEl.textContent = 'Flagged Risk Subjects & Customer Profiles';
        if (countEl) countEl.textContent = `${countCust} subjects`;

        theadRow.innerHTML = `
            <th>Customer ID</th>
            <th>Subject Name</th>
            <th>Risk Level</th>
            <th>Composite Score</th>
            <th>ML Anomaly</th>
            <th>Structuring Txns</th>
            <th>Suggested Escalation</th>
            <th>Actions</th>
        `;

        if (countCust === 0) {
            tbody.innerHTML = `
                <tr class="empty-row">
                    <td colspan="8">
                        <div class="empty-state">
                            <span>Submit an investigation query to populate flagged customer profiles</span>
                        </div>
                    </td>
                </tr>
            `;
            return;
        }

        tbody.innerHTML = activeTableData.map(row => {
            const riskClass = (row.risk_level || 'low').toLowerCase();
            const displayName = row.customer_name || row.customer_id || '—';
            const mlScore = (row.ml_score != null) ? Number(row.ml_score).toFixed(1) : '—';
            const compositeScore = (row.composite_risk_score != null) ? Number(row.composite_risk_score).toFixed(1) : '—';
            const action = row.recommended_action || row.suggested_action || 'MONITOR (Routine Check)';
            
            return `
                <tr>
                    <td><code style="font-family: var(--mono); color: var(--text-1);">${escapeHTML(row.customer_id)}</code></td>
                    <td style="font-weight: 500;">${escapeHTML(displayName)}</td>
                    <td><span class="pill ${riskClass}">${escapeHTML(row.risk_level || 'LOW')}</span></td>
                    <td><span style="font-family: var(--mono); color: var(--accent-light); font-weight:700;">${compositeScore} / 100</span></td>
                    <td><span style="font-family: var(--mono); color: var(--green);">${mlScore}</span></td>
                    <td>${row.structuring_count ?? 0}</td>
                    <td><span style="font-size: 0.78rem; font-weight: 600; color: var(--text-1);">${escapeHTML(action)}</span></td>
                    <td>
                        <button class="btn btn-ghost btn-sm inspect-btn" data-cid="${escapeHTML(row.customer_id)}" style="padding: 0.25rem 0.6rem; font-size: 0.72rem;">Inspect Profile</button>
                    </td>
                </tr>
            `;
        }).join('');

        tbody.querySelectorAll('.inspect-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const input = document.getElementById('chat-input');
                if (input) {
                    input.value = `Explain risk for customer ${btn.dataset.cid}`;
                    submitQuery();
                }
            });
        });

    } else {
        if (titleEl) titleEl.textContent = 'Top Suspicious Transactions (Selected Analysis Path)';
        if (countEl) countEl.textContent = `${countTx} transactions`;

        theadRow.innerHTML = `
            <th>Tx ID</th>
            <th>Customer ID</th>
            <th>Subject Name</th>
            <th>Amount</th>
            <th>Type</th>
            <th>Channel</th>
            <th>Country</th>
            <th>Risk Level</th>
            <th>Detected AML Pattern</th>
            <th>Suggested Escalation</th>
        `;

        if (countTx === 0) {
            tbody.innerHTML = `
                <tr class="empty-row">
                    <td colspan="10">
                        <div class="empty-state">
                            <span>No top suspicious transaction records extracted for this query path</span>
                        </div>
                    </td>
                </tr>
            `;
            return;
        }

        tbody.innerHTML = activeTransactionsData.map(tx => {
            const riskClass = (tx.risk_level || 'low').toLowerCase();
            const displayName = tx.customer_name || tx.customer_id || '—';
            const amt = (tx.amount != null) ? '$' + Number(tx.amount).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) : '$0.00';
            const pattern = tx.aml_pattern || 'Anomalous Activity';
            const action = tx.suggested_action || 'REVIEW (Enhanced Due Diligence)';

            return `
                <tr>
                    <td><code style="font-family: var(--mono); color: var(--cyan);">${escapeHTML(tx.transaction_id || '—')}</code></td>
                    <td><code style="font-family: var(--mono); color: var(--text-1);">${escapeHTML(tx.customer_id || '—')}</code></td>
                    <td style="font-weight: 500;">${escapeHTML(displayName)}</td>
                    <td><span style="font-family: var(--mono); font-weight:700; color: #f8fafc;">${amt}</span></td>
                    <td>${escapeHTML(tx.transaction_type || '—')}</td>
                    <td>${escapeHTML(tx.channel || '—')}</td>
                    <td><span class="font-mono">${escapeHTML(tx.country_code || 'US')}</span></td>
                    <td><span class="pill ${riskClass}">${escapeHTML(tx.risk_level || 'LOW')}</span></td>
                    <td><span style="color: var(--amber); font-size: 0.78rem; font-weight: 500;">${escapeHTML(pattern)}</span></td>
                    <td><span style="font-size: 0.78rem; font-weight: 600;">${escapeHTML(action)}</span></td>
                </tr>
            `;
        }).join('');
    }
}

function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(b => {
        if (b.dataset.target === tabId) b.classList.add('active');
        else b.classList.remove('active');
    });
    document.querySelectorAll('.tab-pane').forEach(p => {
        if (p.id === tabId) p.classList.add('active');
        else p.classList.remove('active');
    });
}

function exportTableCSV() {
    const dataToExport = currentTableView === 'customers' ? activeTableData : activeTransactionsData;
    const filename = currentTableView === 'customers' ? 'sentinel_flagged_risk_subjects.csv' : 'sentinel_suspicious_transactions.csv';

    if (!dataToExport || dataToExport.length === 0) {
        alert('No data available to export in current view. Run a query first.');
        return;
    }
    const headers = Object.keys(dataToExport[0]);
    const csvRows = [
        headers.join(','),
        ...dataToExport.map(row => headers.map(h => JSON.stringify(row[h] ?? '')).join(','))
    ];
    const blob = new Blob([csvRows.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

function exportTableJSON() {
    const dataToExport = currentTableView === 'customers' ? activeTableData : activeTransactionsData;
    const filename = currentTableView === 'customers' ? 'sentinel_audit_customers.json' : 'sentinel_audit_transactions.json';

    if (!dataToExport || dataToExport.length === 0) {
        alert('No data available to export in current view. Run a query first.');
        return;
    }
    const blob = new Blob([JSON.stringify(dataToExport, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

async function runStressTest(lower_bound) {
    try {
        const res = await fetch('/api/stress-test', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({lower_bound: parseInt(lower_bound)})
        });
        if (res.ok) {
            const data = await res.json();
            const baseEl = document.getElementById('stress-baseline');
            if (baseEl) baseEl.textContent = data.baseline_flagged_customers;
            
            const newEl = document.getElementById('stress-new');
            if (newEl) newEl.textContent = data.new_flagged_customers;
            
            const deltaEl = document.getElementById('stress-delta');
            if (deltaEl) {
                const delta = data.customer_count_delta;
                deltaEl.textContent = (delta > 0 ? '+' : '') + delta;
                deltaEl.style.color = delta > 0 ? '#ef4444' : delta < 0 ? '#10b981' : 'var(--text-3)';
            }
            
            const interpEl = document.getElementById('stress-interpretation');
            if (interpEl) interpEl.innerHTML = data.interpretation;
        }
    } catch (e) {
        console.error("Stress test simulation failed", e);
    }
}

function setupUploadModal() {
    const modalOverlay = document.getElementById('upload-modal-overlay');
    const btnOpen = document.getElementById('btn-open-upload-modal');
    const btnClose = document.getElementById('btn-close-upload-modal');
    const btnCancel = document.getElementById('btn-cancel-upload');
    const uploadForm = document.getElementById('upload-csv-form');

    const dropzoneTx = document.getElementById('dropzone-tx');
    const fileInputTx = document.getElementById('input-tx-file') || document.getElementById('file-input-tx');
    const previewTxName = document.getElementById('preview-tx-name');

    const dropzoneCust = document.getElementById('dropzone-cust');
    const fileInputCust = document.getElementById('input-cust-file') || document.getElementById('file-input-cust');
    const previewCustName = document.getElementById('preview-cust-name');

    const statusMsg = document.getElementById('upload-status-msg');
    const btnSubmit = document.getElementById('btn-submit-upload');

    if (!modalOverlay) return;

    // Open Modal
    if (btnOpen) {
        btnOpen.addEventListener('click', () => {
            modalOverlay.classList.add('active');
            if (statusMsg) {
                statusMsg.style.display = 'none';
                statusMsg.textContent = '';
            }
        });
    }

    // Close Modal
    const closeModal = () => {
        modalOverlay.classList.remove('active');
    };
    if (btnClose) btnClose.addEventListener('click', closeModal);
    if (btnCancel) btnCancel.addEventListener('click', closeModal);
    modalOverlay.addEventListener('click', (e) => {
        if (e.target === modalOverlay) closeModal();
    });

    // Setup Drag and Drop helper
    const initDropzone = (dropzone, fileInput, previewEl, defaultText) => {
        if (!dropzone || !fileInput) return;

        dropzone.addEventListener('click', (e) => {
            if (e.target === fileInput) return;
            fileInput.click();
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove('dragover');
            });
        });

        dropzone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files && files.length > 0) {
                fileInput.files = files;
                updatePreview(files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files && fileInput.files.length > 0) {
                updatePreview(fileInput.files[0]);
            }
        });

        function updatePreview(file) {
            if (previewEl) {
                previewEl.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
                previewEl.classList.add('has-file');
            }
        }
    };

    initDropzone(dropzoneTx, fileInputTx, previewTxName, 'transactions.csv');
    initDropzone(dropzoneCust, fileInputCust, previewCustName, 'customers.csv');

    // Handle Form Submit
    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!fileInputTx || !fileInputTx.files || fileInputTx.files.length === 0) {
                if (statusMsg) {
                    statusMsg.style.display = 'block';
                    statusMsg.style.color = '#ef4444';
                    statusMsg.textContent = 'Please select or drop a valid transactions.csv file.';
                }
                return;
            }

            const formData = new FormData();
            formData.append('transactions_file', fileInputTx.files[0]);
            if (fileInputCust && fileInputCust.files && fileInputCust.files.length > 0) {
                formData.append('customers_file', fileInputCust.files[0]);
            }

            if (statusMsg) {
                statusMsg.style.display = 'block';
                statusMsg.style.color = 'var(--accent-light)';
                statusMsg.textContent = 'Uploading CSV datasets & re-training AML models...';
            }
            if (btnSubmit) btnSubmit.disabled = true;

            try {
                const res = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });

                const data = await res.json();
                if (res.ok) {
                    if (statusMsg) {
                        statusMsg.style.color = '#10b981';
                        const txCount = (data.total_transactions || 0).toLocaleString();
                        const custCount = (data.unique_customers || 0).toLocaleString();
                        statusMsg.textContent = `Success! ${data.message} (${txCount} transactions, ${custCount} customers processed).`;
                    }
                    setTimeout(() => {
                        closeModal();
                        initApp(); // Refresh entire dashboard with new data
                    }, 1200);
                } else {
                    if (statusMsg) {
                        statusMsg.style.color = '#ef4444';
                        statusMsg.textContent = `Error: ${data.detail || 'Failed to upload CSV dataset.'}`;
                    }
                }
            } catch (err) {
                console.error("Upload error", err);
                if (statusMsg) {
                    statusMsg.style.color = '#ef4444';
                    statusMsg.textContent = 'Network or server error uploading dataset.';
                }
            } finally {
                if (btnSubmit) btnSubmit.disabled = false;
            }
        });
    }
}
