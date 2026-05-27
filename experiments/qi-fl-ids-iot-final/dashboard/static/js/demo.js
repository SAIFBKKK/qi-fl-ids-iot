(function () {
  const stateUrl = "/api/live-lab/demo-state";
  const refreshMs = 3500;
  const seenDevices = new Set();
  const seenAlerts = new Set();

  function byId(id) {
    return document.getElementById(id);
  }

  function esc(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function fmt(value, fallback = "n/a") {
    if (value === null || value === undefined || value === "") {
      return fallback;
    }
    return esc(value);
  }

  function fmtConfidence(value) {
    if (value === null || value === undefined || value === "") {
      return "n/a";
    }
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return esc(value);
    }
    return number <= 1 ? number.toFixed(3) : number.toFixed(2);
  }

  function statusClass(status) {
    if (status === "ready" || status === "done" || status === "connected") {
      return "badge-success";
    }
    if (status === "degraded" || status === "active" || status === "waiting") {
      return "badge-warning";
    }
    return "badge-danger";
  }

  function severityClass(severity) {
    const normalized = String(severity || "medium").toLowerCase();
    return `severity-${normalized}`;
  }

  function showToast(title, body, kind) {
    if (window.P169Notifications) {
      window.P169Notifications.toast(title, body, { kind });
    }
  }

  function renderHeader(state) {
    const status = String(state.platform_status || "offline").toUpperCase();
    const pill = byId("demo-status-pill");
    if (pill) {
      pill.textContent = `Status: ${status}`;
      pill.className = `status-pill demo-status-${state.platform_status || "offline"}`;
    }
    const refresh = byId("demo-last-refresh");
    if (refresh) {
      refresh.textContent = `Last refresh: ${state.generated_at || "n/a"}`;
    }
  }

  function renderSteps(details) {
    const target = byId("demo-step-timeline");
    if (!target) {
      return;
    }
    target.innerHTML = (details || [])
      .map((step) => `
        <article class="demo-step ${esc(step.status)}">
          <div class="demo-step-top">
            <span>${esc(step.label)}</span>
            <strong class="badge ${statusClass(step.status)}">${esc(step.status)}</strong>
          </div>
          <small>${esc(step.explanation)}</small>
        </article>
      `)
      .join("");
  }

  function renderDevices(devices) {
    const target = byId("demo-device-cards");
    if (!target) {
      return;
    }
    target.innerHTML = (devices || [])
      .map((device) => {
        const connected = device.connected ? "connected" : "waiting";
        const tierOk = device.assigned_tier === device.expected_tier;
        const tierClass = tierOk ? "badge-success" : "badge-warning";
        return `
          <article class="panel demo-device-card ${connected}">
            <div class="demo-card-title">
              <div>
                <p class="eyebrow">Live Device Card</p>
                <h2>${esc(device.node_id)}</h2>
              </div>
              <span class="badge ${statusClass(connected)}">${connected}</span>
            </div>
            <dl class="compact-facts">
              <div><dt>device type</dt><dd>${fmt(device.display_device_type || device.device_type)}</dd></div>
              <div><dt>tier</dt><dd><span class="badge ${tierClass}">${fmt(device.assigned_tier)}</span></dd></div>
              <div><dt>inference path</dt><dd>${fmt(device.inference_path)}</dd></div>
              <div><dt>model</dt><dd>${fmt(device.model_id)}</dd></div>
              <div><dt>QGA mask</dt><dd>${fmt(device.qga_behavior)}</dd></div>
              <div><dt>publish topic</dt><dd><code>${fmt(device.mqtt_publish_topic)}</code></dd></div>
            </dl>
          </article>
        `;
      })
      .join("");
  }

  function renderModel(profile) {
    const model = profile || {};
    const values = {
      "demo-final-model": model.final_model,
      "demo-model-id": model.model_id,
      "demo-mask-id": model.selected_mask_id,
      "demo-threshold": model.threshold,
      "demo-scaler": model.scaler,
      "demo-vm1-path": model.vm1_path,
      "demo-vm2-path": model.vm2_path,
    };
    Object.entries(values).forEach(([id, value]) => {
      const node = byId(id);
      if (node) {
        node.textContent = value ?? "n/a";
      }
    });
  }

  function renderLatestAlert(alert) {
    const panel = byId("demo-alert-focus");
    const badge = byId("demo-alert-severity");
    const target = byId("demo-latest-alert");
    if (!panel || !badge || !target) {
      return;
    }
    if (!alert) {
      panel.className = "panel demo-alert-focus severity-medium";
      badge.className = "badge severity-medium";
      badge.textContent = "waiting";
      target.innerHTML = `
        <strong>No alert detected yet.</strong>
        <p>Publish a controlled PacketWindow(30) flow from VM1 or VM2 to show the live IDS alert path.</p>
      `;
      return;
    }
    const severity = String(alert.severity || "medium").toLowerCase();
    panel.className = `panel demo-alert-focus ${severityClass(severity)}`;
    badge.className = `badge ${severityClass(severity)}`;
    badge.textContent = severity;
    target.innerHTML = `
      <div class="alert-focus-title">ALERT DETECTED</div>
      <dl class="compact-facts alert-focus-facts">
        <div><dt>node_id</dt><dd>${fmt(alert.node_id)}</dd></div>
        <div><dt>severity</dt><dd>${fmt(severity)}</dd></div>
        <div><dt>label</dt><dd>${fmt(alert.predicted_label)}</dd></div>
        <div><dt>confidence</dt><dd>${fmtConfidence(alert.confidence)}</dd></div>
        <div><dt>flow_id</dt><dd>${fmt(alert.flow_id)}</dd></div>
        <div><dt>timestamp</dt><dd>${fmt(alert.timestamp)}</dd></div>
        <div><dt>source topic</dt><dd><code>${fmt(alert.source_topic)}</code></dd></div>
      </dl>
    `;
  }

  function renderMetrics(metrics) {
    const target = byId("demo-metrics-grid");
    if (!target) {
      return;
    }
    const nodes = metrics?.nodes || {};
    const errors = metrics?.errors || {};
    const items = [];
    Object.entries(nodes).forEach(([nodeId, counts]) => {
      items.push(["Flows", nodeId, counts.flows, "badge-info"]);
      items.push(["Predictions", nodeId, counts.predictions, "badge-success"]);
      items.push(["Alerts", nodeId, counts.alerts, "badge-danger"]);
    });
    items.push(["API errors", "final-ids-api", errors.final_ids_api_prediction_errors_total, errors.final_ids_api_prediction_errors_total === 0 ? "badge-success" : "badge-danger"]);
    items.push(["Bridge errors", "final-mqtt-bridge", errors.final_mqtt_bridge_prediction_errors_total, errors.final_mqtt_bridge_prediction_errors_total === 0 ? "badge-success" : "badge-danger"]);
    target.innerHTML = items
      .map(([label, scope, value, klass]) => `
        <article class="metric-chip">
          <span>${esc(label)}</span>
          <strong>${fmt(value, "0")}</strong>
          <small>${esc(scope)}</small>
          <em class="badge ${klass}">${Number(value || 0) === 0 && label.includes("errors") ? "zero" : "live"}</em>
        </article>
      `)
      .join("");
  }

  function renderEvents(events) {
    const target = byId("demo-event-stream");
    if (!target) {
      return;
    }
    if (!events || !events.length) {
      target.className = "demo-event-stream empty-state";
      target.textContent = "No live events observed yet.";
      return;
    }
    target.className = "demo-event-stream";
    target.innerHTML = events
      .map((event) => `
        <article class="demo-event ${severityClass(event.severity || "info")}">
          <div>
            <strong>${esc(event.title || event.type)}</strong>
            <p>${esc(event.detail || "")}</p>
          </div>
          <small>${fmt(event.node_id)}<br>${fmt(event.timestamp)}</small>
        </article>
      `)
      .join("");
  }

  function renderServices(services) {
    const target = byId("demo-service-status");
    if (!target) {
      return;
    }
    target.innerHTML = Object.entries(services || {})
      .map(([name, service]) => `
        <div class="service-item">
          <div><strong>${esc(name)}</strong><span>${service.ok ? "reachable" : "unavailable"}</span></div>
          <span class="badge ${service.ok ? "badge-success" : "badge-danger"}">${service.ok ? "ready" : "offline"}</span>
        </div>
      `)
      .join("");
  }

  function renderWarnings(warnings) {
    const target = byId("demo-warnings");
    if (!target) {
      return;
    }
    target.innerHTML = (warnings || [])
      .map((warning) => `<p class="warning-line">${esc(warning)}</p>`)
      .join("");
  }

  function detectNotifications(state) {
    (state.devices || []).forEach((device) => {
      if (!device.connected) {
        return;
      }
      const key = `device:${device.node_id}`;
      if (!seenDevices.has(key)) {
        seenDevices.add(key);
        showToast(
          `Device connected: ${device.node_id}`,
          `${device.device_type || device.display_device_type}, tier ${device.assigned_tier}, model ${device.model_id}, mask ${device.selected_mask_id}`,
          "device"
        );
      }
    });
    const alert = state.latest_alert;
    if (alert) {
      const key = `alert:${alert.node_id}:${alert.flow_id}:${alert.timestamp}`;
      if (!seenAlerts.has(key)) {
        seenAlerts.add(key);
        showToast(
          `Alert detected on ${alert.node_id}`,
          `${alert.severity || "medium"} severity, label ${alert.predicted_label || "unknown"}, confidence ${fmtConfidence(alert.confidence)}, flow ${alert.flow_id || "n/a"}`,
          "alert"
        );
      }
    }
  }

  function render(state) {
    renderHeader(state);
    renderSteps(state.step_details);
    renderDevices(state.devices);
    renderModel(state.model_profile);
    renderLatestAlert(state.latest_alert);
    renderMetrics(state.metrics);
    renderEvents(state.recent_events);
    renderServices(state.services);
    renderWarnings(state.warnings);
    detectNotifications(state);
  }

  async function refresh() {
    try {
      const response = await fetch(stateUrl, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      render(await response.json());
    } catch (error) {
      const pill = byId("demo-status-pill");
      if (pill) {
        pill.textContent = "Status: OFFLINE";
        pill.className = "status-pill demo-status-offline";
      }
      renderWarnings([`Demo state unavailable: ${error}`]);
    }
  }

  function setupControls() {
    const refreshButton = byId("demo-refresh");
    if (refreshButton) {
      refreshButton.addEventListener("click", refresh);
    }
    const clearButton = byId("demo-clear-notifications");
    if (clearButton) {
      clearButton.addEventListener("click", () => {
        const stack = byId("toast-stack");
        if (stack) {
          stack.innerHTML = "";
        }
      });
    }
    const focusButton = byId("demo-focus-alert");
    if (focusButton) {
      focusButton.addEventListener("click", () => {
        const panel = byId("demo-alert-focus");
        if (panel) {
          panel.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      });
    }
    const toggleTheme = byId("demo-toggle-theme");
    if (toggleTheme) {
      toggleTheme.addEventListener("click", () => {
        const button = byId("theme-toggle");
        if (button) {
          button.click();
        }
      });
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    setupControls();
    refresh();
    window.setInterval(refresh, refreshMs);
  });
})();
