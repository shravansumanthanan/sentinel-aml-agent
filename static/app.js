document.addEventListener("DOMContentLoaded", () => {
    fetchDatasetSummary();
});

let currentSarText = "";

async function fetchDatasetSummary() {
    try {
        const res = await fetch("/api/dataset/summary");
        if (res.ok) {
            const data = await res.json();
            const sum = data.summary;
            document.getElementById("metric-volume").innerText = `$${sum.total_volume.toLocaleString('en-US', {minimumFractionDigits: 2})}`;
            document.getElementById("metric-tx-count").innerText = sum.total_transactions.toLocaleString();
            document.getElementById("metric-customers").innerText = sum.unique_customers.toLocaleString();
        }
    } catch (err) {
        console.error("Failed to load dataset summary:", err);
    }
}

function switchTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach(pane => pane.classList.remove("active"));

    const activeBtn = Array.from(document.querySelectorAll(".tab-btn")).find(b => b.getAttribute("onclick").includes(tabId));
    if (activeBtn) activeBtn.classList.add("active");

    const activePane = document.getElementById(`pane-${tabId}`);
    if (activePane) activePane.classList.add("active");
}

function sendSampleQuery(text) {
    document.getElementById("chat-input").value = text;
    handleChatSubmit(new Event("submit"));
}

async function handleChatSubmit(e) {
    e.preventDefault();
    const inputEl = document.getElementById("chat-input");
    const query = inputEl.value.trim();
    if (!query) return;

    // Add User Message to Chat
    appendChatMessage("user", query);
    inputEl.value = "";

    // Show Agent Thinking
    const thinkingId = appendChatMessage("agent", "🤖 <em>Orchestrating agent execution plan & invoking tools...</em>");

    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: query })
        });

        if (!res.ok) throw new Error("API Error");

        const data = await res.json();
        
        // Remove thinking message
        const thinkingNode = document.getElementById(thinkingId);
        if (thinkingNode) thinkingNode.remove();

        // Format agent natural language explanation
        let agentText = "<strong>Agent Decision Summary:</strong><br>";
        if (data.explanations && data.explanations.length > 0) {
            agentText += data.explanations.join("<br><br>");
        } else {
            agentText += "Query executed successfully. View Telemetry and Risk Workbench for details.";
        }

        appendChatMessage("agent", agentText);

        // Render Telemetry & Workbench Data
        renderTelemetry(data.parsed_intent, data.extracted_entities, data.telemetry);
        renderTableData(data.results);
        renderSar(data.sar_narrative);

    } catch (err) {
        console.error(err);
        appendChatMessage("agent", "⚠️ Error executing query. Please try again.");
    }
}

function appendChatMessage(sender, text) {
    const chatBox = document.getElementById("chat-messages");
    const msgDiv = document.createElement("div");
    const msgId = "msg-" + Date.now();
    msgDiv.id = msgId;
    msgDiv.className = `message ${sender}-msg`;

    const avatar = sender === "user" ? "👤" : "🤖";
    msgDiv.innerHTML = `
        <div class="msg-avatar">${avatar}</div>
        <div class="msg-content">${text}</div>
    `;

    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return msgId;
}

function renderTelemetry(intent, entities, telemetry) {
    document.getElementById("telemetry-intent").innerText = intent || "GENERAL";
    document.getElementById("telemetry-entities").innerText = JSON.stringify(entities, null, 2);

    const planList = document.getElementById("telemetry-plan");
    planList.innerHTML = "";
    if (telemetry.execution_plan && telemetry.execution_plan.length > 0) {
        telemetry.execution_plan.forEach(step => {
            const li = document.createElement("li");
            li.innerText = step;
            planList.appendChild(li);
        });
    } else {
        planList.innerHTML = "<li>No specific steps required.</li>";
    }

    document.getElementById("telemetry-called").innerText = (telemetry.tools_called || []).join(", ") || "None";
    document.getElementById("telemetry-skipped").innerText = (telemetry.tools_skipped || []).join(", ") || "None";
    
    if (telemetry.latency_ms) {
        document.getElementById("metric-latency").innerText = `${telemetry.latency_ms} ms`;
    }
}

function renderTableData(results) {
    const tbody = document.getElementById("table-body");
    tbody.innerHTML = "";

    let rows = [];
    if (results.flagged_table) {
        rows = results.flagged_table;
    } else if (results.single_lookup && results.single_lookup.found) {
        const cust = results.single_lookup.customer;
        const risk = results.single_lookup.risk_profile;
        rows = [{ ...cust, ...risk }];
    }

    document.getElementById("table-count").innerText = `${rows.length} Subjects Listed`;

    if (rows.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" class="empty-state">No matching subjects found for this query path.</td></tr>`;
        return;
    }

    rows.forEach(r => {
        const tr = document.createElement("tr");
        const riskClass = (r.risk_level || "LOW").toLowerCase();
        
        tr.innerHTML = `
            <td><strong>${r.customer_id}</strong></td>
            <td>${r.customer_name || 'N/A'}</td>
            <td>${r.risk_rating || 'Low'}</td>
            <td><strong>${r.composite_risk_score || 0}</strong>/100</td>
            <td>${r.structuring_count || 0}</td>
            <td><span class="risk-pill ${riskClass}">${r.recommended_action || 'MONITOR'}</span></td>
            <td><button class="action-btn" onclick="inspectCustomer('${r.customer_id}')">Inspect SAR</button></td>
        `;
        tbody.appendChild(tr);
    });
}

function renderSar(sarText) {
    const textarea = document.getElementById("sar-textarea");
    if (sarText) {
        currentSarText = sarText;
        textarea.value = sarText;
    } else if (!currentSarText) {
        textarea.value = "No High Risk SAR narrative generated for current query.";
    }
}

function copySarNarrative() {
    const textarea = document.getElementById("sar-textarea");
    if (!textarea.value) return;
    navigator.clipboard.writeText(textarea.value);
    alert("SAR Narrative copied to clipboard!");
}

function inspectCustomer(cid) {
    sendSampleQuery(`Is customer ID ${cid} suspicious?`);
    switchTab('sar');
}

function updateSliderLabel(val) {
    document.getElementById("slider-val").innerText = `$${parseInt(val).toLocaleString()}`;
}

async function runStressTest(val) {
    try {
        const res = await fetch("/api/stress-test", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ lower_bound: parseFloat(val) })
        });
        if (res.ok) {
            const data = await res.json();
            document.getElementById("stress-baseline").innerText = data.baseline_flagged_customers;
            document.getElementById("stress-new").innerText = data.new_flagged_customers;
            document.getElementById("stress-delta").innerText = `+${data.customer_count_delta}`;
            document.getElementById("stress-interpretation").innerText = data.interpretation;
        }
    } catch (err) {
        console.error(err);
    }
}
