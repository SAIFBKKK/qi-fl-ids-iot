(function () {
  const pollMs = 2500;
  const knownNodes = new Set();
  const knownAlerts = new Set();
  const deviceEvents = [];
  let firstPoll = true;

  function qs(id) {
    return document.getElementById(id);
  }

  function esc(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      "\"": "&quot;",
      "'": "&#39;",
    }[char]));
  }

  function number(value) {
    return Number(value || 0).toLocaleString();
  }

  function fmtTime(value) {
    if (!value) {
      return "n/a";
    }
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) {
      return esc(value);
    }
    return parsed.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function setText(id, value) {
    const element = qs(id);
    if (element) {
      element.textContent = value;
    }
  }

  function badge(text, cls = "badge-info") {
    return `<span class="badge ${cls}">${esc(text)}</span>`;
  }

  function severityClass(severity) {
    const value = String(severity || "medium").toLowerCase();
    if (["low", "medium", "high", "critical"].includes(value)) {
      return `severity-${value}`;
    }
    return "severity-medium";
  }

  function alertLabel(alert) {
    return alert.predicted_label || alert.label || alert.prediction_label || "unknown";
  }

  function alertConfidence(alert) {
    const value = alert.confidence ?? alert.probability_attack ?? alert.attack_probability ?? alert.score;
    if (value === null || value === undefined || value === "") {
      return "n/a";
    }
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed.toFixed(3) : String(value);
  }

  function renderKpis(state) {
    const kpis = state.kpis || {};
    setText("kpi-devices", number(kpis.connected_devices));
    setText("kpi-flows", number(kpis.flows_observed));
    setText("kpi-predictions", number(kpis.predictions));
    setText("kpi-alerts", number(kpis.alerts));
    setText("kpi-api-errors", number(kpis.api_errors));
    setText("kpi-bridge-errors", number(kpis.bridge_errors));
    setText("last-refresh", `Last refresh ${fmtTime(state.generated_at)}`);

    const status = (state.platform && state.platform.status) || "degraded";
    const pill = qs("platform-status-pill");
    if (pill) {
      pill.className = `status-pill status-${status}`;
      pill.textContent = `Platform: ${status}`;
    }
  }

  function renderDevices(nodes) {
    const body = qs("device-table-body");
    if (!body) {
      return;
    }
    if (!nodes || nodes.length === 0) {
      body.innerHTML = '<tr><td colspan="12" class="empty-row">No live lab nodes registered yet.</td></tr>';
      return;
    }
    body.innerHTML = nodes.map((node) => {
      const modes = (node.supported_input_modes || []).map((mode) => `<code>${esc(mode)}</code>`).join("");
      return `
        <tr>
          <td>${badge(node.status || "connected", "badge-success")}</td>
          <td><strong>${esc(node.node_id)}</strong></td>
          <td>${esc(node.hostname)}</td>
          <td>${esc(node.device_type)}</td>
          <td>${esc(node.cpu_count)}</td>
          <td>${esc(node.ram_gb)} GB</td>
          <td>${badge(node.assigned_tier || "unknown", node.assigned_tier === "medium" ? "badge-info" : "badge-warning")}</td>
          <td><code>${esc(node.model_id)}</code></td>
          <td><code>${esc(node.selected_mask_id)}</code></td>
          <td><div class="mode-stack">${modes}</div></td>
          <td><div class="topic-stack"><code>${esc(node.mqtt_publish_topic)}</code><code>${esc(node.mqtt_prediction_topic)}</code><code>${esc(node.mqtt_alert_topic)}</code></div></td>
          <td>${fmtTime(node.registered_at)}<br><span class="muted">updated ${fmtTime(node.updated_at)}</span></td>
        </tr>
      `;
    }).join("");
  }

  function renderModel(profile) {
    const model = profile || {};
    setText("model-id", model.model_id || "p8_fedavg_qga_l1");
    setText("selected-mask-id", model.selected_mask_id || "conservative_seed_42");
    setText("model-threshold", model.threshold ?? "runtime default");
    setText("model-input-modes", (model.supported_input_modes || []).join(", "));
  }

  function renderServices(services) {
    const target = qs("service-status-list");
    if (!target) {
      return;
    }
    const labels = { controller: "live-lab-controller", validator: "online-validator", bridge: "final-mqtt-bridge", api: "final-ids-api" };
    target.innerHTML = Object.entries(labels).map(([key, label]) => {
      const service = services && services[key] ? services[key] : {};
      return `
        <div class="service-item">
          <div><strong>${label}</strong><br><span>${esc((service.ready && service.ready.url) || (service.health && service.health.url) || "")}</span></div>
          ${badge(service.ok ? "online" : "offline", service.ok ? "badge-success" : "badge-danger")}
        </div>
      `;
    }).join("");
  }

  function renderTopicCounts(topicCounts) {
    const target = qs("topic-counts");
    if (!target) {
      return;
    }
    const entries = Object.entries(topicCounts || {}).filter(([topic]) => topic.startsWith("ids/")).slice(0, 12);
    if (entries.length === 0) {
      target.className = "topic-counts empty-state";
      target.textContent = "No topic counts available.";
      return;
    }
    target.className = "topic-counts";
    target.innerHTML = entries.map(([topic, count]) => `
      <div class="topic-item"><code>${esc(topic)}</code><strong>${number(count)}</strong></div>
    `).join("");
  }

  function alertKey(alert) {
    return `${alert.source_topic}|${alert.flow_id}|${alert.timestamp}|${alert.received_at_unix}`;
  }

  function detectNodeNotifications(nodes) {
    (nodes || []).forEach((node) => {
      if (knownNodes.has(node.node_id)) {
        return;
      }
      knownNodes.add(node.node_id);
      const event = {
        node_id: node.node_id,
        device_type: node.device_type,
        assigned_tier: node.assigned_tier,
        model_id: node.model_id,
        selected_mask_id: node.selected_mask_id,
        timestamp: new Date().toISOString(),
      };
      deviceEvents.unshift(event);
      deviceEvents.splice(8);
      P169Notifications.toast(
        `Device connected: ${node.node_id}`,
        `${node.device_type || "unknown"} assigned to ${node.assigned_tier || "unknown"} with ${node.model_id || "model"}`
      );
    });
  }

  function renderDeviceEvents() {
    const target = qs("device-event-list");
    if (!target) {
      return;
    }
    if (deviceEvents.length === 0) {
      target.className = "event-list empty-state";
      target.textContent = "No device events yet.";
      return;
    }
    target.className = "event-list";
    target.innerHTML = deviceEvents.map((event) => `
      <div class="event-item">
        <strong>Device connected: ${esc(event.node_id)}</strong>
        <div class="event-meta">
          <span>${esc(event.device_type)}</span>
          <span>${esc(event.assigned_tier)}</span>
          <span>${esc(event.model_id)}</span>
          <span>${esc(event.selected_mask_id)}</span>
          <span>${fmtTime(event.timestamp)}</span>
        </div>
      </div>
    `).join("");
  }

  function detectAlertNotifications(alerts) {
    (alerts || []).slice(0, firstPoll ? 2 : alerts.length).forEach((alert) => {
      const key = alertKey(alert);
      if (knownAlerts.has(key)) {
        return;
      }
      knownAlerts.add(key);
      const label = alertLabel(alert);
      const confidence = alertConfidence(alert);
      P169Notifications.toast(
        `Alert detected on ${alert.node_id}`,
        `${alert.severity || "medium"} severity, label ${label}, confidence ${confidence}, flow ${alert.flow_id || "n/a"}`,
        { kind: "alert", timeout: 6500 }
      );
    });
  }

  function renderAlerts(alerts) {
    const target = qs("alert-event-list");
    if (!target) {
      return;
    }
    if (!alerts || alerts.length === 0) {
      target.className = "alert-list empty-state";
      target.textContent = "No IDS alerts observed yet.";
      return;
    }
    target.className = "alert-list";
    target.innerHTML = alerts.map((alert) => {
      const cls = severityClass(alert.severity);
      const label = alertLabel(alert);
      const confidence = alertConfidence(alert);
      return `
        <div class="alert-item ${cls}">
          <strong>Alert detected on ${esc(alert.node_id)}</strong>
          <div class="alert-meta">
            ${badge(alert.severity || "medium", cls)}
            <span>label ${esc(label)}</span>
            <span>confidence ${esc(confidence)}</span>
            <span>flow ${esc(alert.flow_id || "n/a")}</span>
            <span>${fmtTime(alert.timestamp)}</span>
          </div>
          <code>${esc(alert.source_topic)}</code>
        </div>
      `;
    }).join("");
  }

  function render(state) {
    renderKpis(state);
    renderDevices(state.nodes || []);
    renderModel(state.model_profile || {});
    renderServices(state.services || {});
    renderTopicCounts(state.topic_counts || {});
    detectNodeNotifications(state.nodes || []);
    renderDeviceEvents();
    detectAlertNotifications(state.recent_alerts || []);
    renderAlerts(state.recent_alerts || []);
    firstPoll = false;
  }

  async function poll() {
    try {
      const response = await fetch("/api/live-lab/state", { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      render(await response.json());
    } catch (error) {
      const pill = qs("platform-status-pill");
      if (pill) {
        pill.className = "status-pill status-degraded";
        pill.textContent = "Platform: dashboard polling error";
      }
      console.error("P16.9 dashboard polling failed", error);
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    poll();
    window.setInterval(poll, pollMs);
  });
})();
