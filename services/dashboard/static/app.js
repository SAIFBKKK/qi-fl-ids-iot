const MODE_A_NODE_IDS = ['node1', 'node2', 'node3'];

const MODEL_FALLBACKS = {
  weak: { hidden_dims: [64], md5: '2a180230', param_count: 4066 },
  medium: { hidden_dims: [128, 64], md5: '04718ffb', param_count: 14178 },
  powerful: { hidden_dims: [256, 128], md5: '3c0452e8', param_count: 44706 },
};

const NODE_FALLBACKS = {
  node1: { assigned_tier: 'weak', cpu_cores: 1, ram_mb: 512, device_type: 'raspberrypi-zero' },
  node2: { assigned_tier: 'medium', cpu_cores: 2, ram_mb: 1024, device_type: 'raspberrypi-3' },
  node3: { assigned_tier: 'powerful', cpu_cores: 4, ram_mb: 2048, device_type: 'raspberrypi-4' },
};

function nowTime() {
  return new Date().toLocaleTimeString('fr-FR', { hour12: false });
}

function readPromValue(payload) {
  const raw = payload?.data?.result?.[0]?.value?.[1];
  if (raw === undefined || raw === null) return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

function readPromVector(payload) {
  const rows = payload?.data?.result || [];
  return rows.map((row) => ({
    metric: row.metric || {},
    value: Number(row.value?.[1]),
  })).filter((row) => Number.isFinite(row.value));
}

async function promQuery(query) {
  const response = await fetch(`/api/prometheus/query?q=${encodeURIComponent(query)}`);
  if (!response.ok) throw new Error(`prometheus ${response.status}`);
  return response.json();
}

function formatCountValue(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'N/A';
  return new Intl.NumberFormat('fr-FR', { maximumFractionDigits: 0 }).format(Number(value));
}

function formatFloatValue(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'N/A';
  return Number(value).toFixed(digits);
}

function formatMsValue(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'N/A';
  return `${Number(value).toFixed(1)} ms`;
}

function formatPercentValue(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'N/A';
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function formatDateValue(value) {
  if (!value) return null;
  const date = typeof value === 'number' ? new Date(value) : new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleString('fr-FR', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function statusClass(status) {
  if (status === 'measured') return 'badge-measured';
  if (status === 'expected' || status === 'planned') return 'badge-expected';
  if (status === 'literature') return 'badge-literature';
  if (status === 'mock') return 'badge-mock';
  if (status === 'replay') return 'badge-replay';
  return 'badge-na';
}

function headerStatus() {
  return {
    coreOverall: 'unknown',
    coreLabel: '0/4',
    coreTitle: 'Chargement /api/system/health',
    idsOverall: 'unknown',
    idsLabel: '0/3',
    idsTitle: 'Chargement /api/nodes',
    modeLabel: 'LIVE',
    scenarioLabel: 'deployment_15.parquet',
    clock: nowTime(),

    async init() {
      await this.refresh();
      setInterval(() => {
        this.clock = nowTime();
      }, 1000);
      setInterval(() => this.refresh(), 10000);
      window.addEventListener('mode-change', (event) => {
        const { mode, scenario } = event.detail || {};
        this.modeLabel = mode === 'replay' ? 'REPLAY' : 'LIVE';
        this.scenarioLabel = scenario?.name || 'deployment_15.parquet';
      });
    },

    async refresh() {
      await Promise.all([this.refreshCore(), this.refreshIds()]);
    },

    async refreshCore() {
      try {
        const response = await fetch('/api/system/health');
        if (!response.ok) throw new Error(`system ${response.status}`);
        const data = await response.json();
        this.coreOverall = data.overall === 'healthy' ? 'healthy' : data.overall === 'degraded' ? 'degraded' : 'down';
        this.coreLabel = `${data.ups ?? 0}/${data.total ?? 4}`;
        this.coreTitle = (data.services || []).map((service) => (
          `${service.service}: ${service.status}${service.latency_ms ? ` (${service.latency_ms} ms)` : ''}`
        )).join('\n');
      } catch (err) {
        this.coreOverall = 'down';
        this.coreLabel = 'N/A';
        this.coreTitle = `Erreur /api/system/health: ${err.message}`;
      }
    },

    async refreshIds() {
      try {
        const response = await fetch('/api/nodes');
        if (!response.ok) throw new Error(`nodes ${response.status}`);
        const data = await response.json();
        const nodes = (data.nodes || []).filter((node) => MODE_A_NODE_IDS.includes(node.node_id));
        let online = nodes.filter((node) => ['connected', 'registered', 'detecting'].includes(node.status)).length;
        const total = nodes.length || 3;
        if (nodes.length === 0) {
          const promNodes = await promQuery('sum(ids_node_status{node_id=~"node1|node2|node3"})');
          online = readPromValue(promNodes) || 0;
        }
        this.idsOverall = online === total ? 'healthy' : online > 0 ? 'degraded' : 'down';
        this.idsLabel = `${online}/${total}`;
        this.idsTitle = nodes.length
          ? nodes.map((node) => `${node.node_id}: ${node.status} (${node.assigned_tier || 'tier N/A'})`).join('\n')
          : 'Fallback Prometheus: ids_node_status for node1, node2, node3';
      } catch (err) {
        this.idsOverall = 'down';
        this.idsLabel = 'N/A';
        this.idsTitle = `Erreur /api/nodes: ${err.message}`;
      }
    },
  };
}

function iotTab() {
  return {
    nodes: [],
    modelsByTier: {},
    selected: null,
    busy: false,
    error: null,
    sparklines: {},
    kpis: {
      flows: null,
      predictions: null,
      alerts: null,
      avgLatencyMs: null,
      rejectRate: null,
      generatorPublished: null,
    },
    targetDistribution: [],
    attackDistribution: [],
    replayAlerts: [],
    mqttConnected: null,
    prometheusUp: null,
    lastUpdated: null,
    interval: null,

    async init() {
      window.addEventListener('scenario-event', (event) => {
        const detail = event.detail || {};
        if (detail.type === 'alert') {
          this.replayAlerts.unshift({
            ...detail,
            key: `${detail.t}-${detail.node_id}-${detail.attack_class}-${this.replayAlerts.length}`,
          });
          this.replayAlerts = this.replayAlerts.slice(0, 8);
        }
      });
      window.addEventListener('mode-change', (event) => {
        if ((event.detail || {}).mode !== 'replay') this.replayAlerts = [];
      });
      await this.loadAll();
      this.interval = setInterval(() => this.loadAll(), 5000);
    },

    async loadAll() {
      await Promise.all([this.loadNodes(), this.loadMetrics()]);
      this.lastUpdated = new Date();
    },

    async loadNodes() {
      try {
        const [nodesResp, modelsResp] = await Promise.all([
          fetch('/api/nodes'),
          fetch('/api/models'),
        ]);
        if (!nodesResp.ok) throw new Error(`nodes ${nodesResp.status}`);
        if (!modelsResp.ok) throw new Error(`models ${modelsResp.status}`);

        const nodesData = await nodesResp.json();
        const modelsData = await modelsResp.json();
        const modelRows = modelsData.tiers || modelsData.models || [];
        this.modelsByTier = {};
        modelRows.forEach((tier) => {
          const fallback = MODEL_FALLBACKS[tier.tier] || {};
          this.modelsByTier[tier.tier] = {
            tier: tier.tier,
            hidden_dims: tier.config?.hidden_dims || fallback.hidden_dims || [],
            md5: tier.md5 || fallback.md5 || 'N/A',
            param_count: fallback.param_count || null,
            size_bytes: tier.size_bytes ?? null,
            config: tier.config || {},
          };
        });

        let nodes = (nodesData.nodes || []);
        if (!nodes.some((node) => MODE_A_NODE_IDS.includes(node.node_id))) {
          nodes = await this.loadPrometheusNodeInventory();
        }

        this.nodes = nodes
          .map((node) => ({
            ...node,
            metrics: node.metrics || {},
          }))
          .sort((a, b) => a.node_id.localeCompare(b.node_id));

        await this.loadNodeMetrics();
        this.error = null;
        setTimeout(() => {
          this.modeANodes.forEach((node) => this.updateSparkline(node.node_id));
        }, 0);
      } catch (err) {
        this.error = `Dashboard indisponible: ${err.message}`;
        console.error('loadNodes failed:', err);
      }
    },

    async loadPrometheusNodeInventory() {
      const statusRows = {};
      const tierRows = {};
      try {
        const [statusPayload, tierPayload] = await Promise.all([
          promQuery('ids_node_status{node_id=~"node1|node2|node3"}'),
          promQuery('ids_node_assigned_tier_info{node_id=~"node1|node2|node3"}'),
        ]);
        readPromVector(statusPayload).forEach((row) => {
          if (row.metric.node_id) statusRows[row.metric.node_id] = row.value;
        });
        readPromVector(tierPayload).forEach((row) => {
          if (row.metric.node_id && row.metric.tier && row.value >= 1) {
            tierRows[row.metric.node_id] = row.metric.tier;
          }
        });
      } catch (err) {
        console.warn('Prometheus node inventory fallback failed:', err);
      }

      return MODE_A_NODE_IDS.map((nodeId) => {
        const fallback = NODE_FALLBACKS[nodeId] || {};
        const up = statusRows[nodeId];
        return {
          node_id: nodeId,
          cpu_cores: fallback.cpu_cores || null,
          ram_mb: fallback.ram_mb || null,
          device_type: fallback.device_type || 'docker_node',
          network_quality: nodeId === 'node1' ? 'low' : nodeId === 'node2' ? 'medium' : 'high',
          battery_powered: nodeId === 'node1',
          assigned_tier: tierRows[nodeId] || fallback.assigned_tier || 'unknown',
          model_version: 'model_factory_30rounds',
          model_source: 'prometheus_fallback',
          status: up === undefined ? 'connected' : up >= 1 ? 'connected' : 'disconnected',
          registered_at: null,
          updated_at: null,
          last_heartbeat: null,
        };
      });
    },

    async loadNodeMetrics() {
      await Promise.all(this.nodes.map(async (node) => {
        if (!MODE_A_NODE_IDS.includes(node.node_id)) return;
        const nodeId = node.node_id;
        try {
          const [flows, predictions, alerts, latency] = await Promise.all([
            promQuery(`sum(ids_flows_received_total{node_id="${nodeId}"})`),
            promQuery(`sum(ids_predictions_total{node_id="${nodeId}"})`),
            promQuery(`sum(ids_alerts_total{node_id="${nodeId}"})`),
            promQuery(`sum(rate(inference_latency_seconds_sum{node_id="${nodeId}"}[1m])) / sum(rate(inference_latency_seconds_count{node_id="${nodeId}"}[1m])) * 1000`),
          ]);
          node.metrics = {
            flows: readPromValue(flows),
            predictions: readPromValue(predictions),
            alerts: readPromValue(alerts),
            avgLatencyMs: readPromValue(latency),
          };
        } catch (_err) {
          node.metrics = {
            flows: null,
            predictions: null,
            alerts: null,
            avgLatencyMs: null,
          };
        }
      }));
    },

    async loadMetrics() {
      try {
        const [
          flows,
          predictions,
          alerts,
          latency,
          rejectRate,
          generatorPublished,
          mqttConnected,
          up,
          byTarget,
          byAttack,
        ] = await Promise.all([
          promQuery('sum(ids_flows_received_total)'),
          promQuery('sum(ids_predictions_total)'),
          promQuery('sum(ids_alerts_total)'),
          promQuery('sum(rate(inference_latency_seconds_sum[1m])) / sum(rate(inference_latency_seconds_count[1m])) * 1000'),
          promQuery('sum(rate(ids_flows_rejected_invalid_schema_total[1m])) / sum(rate(ids_flows_received_total[1m]))'),
          promQuery('sum(traffic_generator_flows_published_total)'),
          promQuery('max(traffic_generator_mqtt_connected)'),
          promQuery('min(up)'),
          promQuery('sum by (target_node_id) (traffic_generator_flows_published_by_target_total)'),
          promQuery('sum by (predicted_label) (ids_predictions_total)'),
        ]);

        this.kpis.flows = readPromValue(flows);
        this.kpis.predictions = readPromValue(predictions);
        this.kpis.alerts = readPromValue(alerts);
        this.kpis.avgLatencyMs = readPromValue(latency);
        this.kpis.rejectRate = readPromValue(rejectRate);
        this.kpis.generatorPublished = readPromValue(generatorPublished);
        this.mqttConnected = readPromValue(mqttConnected);
        this.prometheusUp = readPromValue(up);
        this.targetDistribution = this.vectorToBars(readPromVector(byTarget), 'target_node_id');
        this.attackDistribution = this.vectorToBars(readPromVector(byAttack), 'predicted_label')
          .filter((row) => row.value > 0)
          .slice(0, 10);
      } catch (err) {
        console.error('loadMetrics failed:', err);
      }
    },

    vectorToBars(rows, labelName) {
      const max = Math.max(...rows.map((row) => row.value), 0);
      return rows
        .map((row) => ({
          label: row.metric[labelName] || 'unknown',
          value: row.value,
          percent: max > 0 ? Math.max(4, Math.round((row.value / max) * 100)) : 0,
        }))
        .sort((a, b) => b.value - a.value);
    },

    get modeANodes() {
      const byId = {};
      this.nodes.forEach((node) => {
        byId[node.node_id] = node;
      });
      return MODE_A_NODE_IDS.map((id) => byId[id]).filter(Boolean);
    },

    get modeAOnline() {
      return this.modeANodes.filter((node) => this.isConnected(node)).length;
    },

    get lastUpdatedLabel() {
      return this.lastUpdated ? this.lastUpdated.toLocaleTimeString('fr-FR', { hour12: false }) : 'N/A';
    },

    get mqttConnectedLabel() {
      if (this.mqttConnected === null || this.mqttConnected === undefined) return 'N/A';
      return this.mqttConnected >= 1 ? 'connected' : 'down';
    },

    get prometheusLabel() {
      if (this.prometheusUp === null || this.prometheusUp === undefined) return 'N/A';
      return this.prometheusUp >= 1 ? 'scraping' : 'degraded';
    },

    tierOf(node) {
      return node.assigned_tier || MODEL_FALLBACKS[node.node_id] || 'unknown';
    },

    modelMeta(node) {
      const tier = this.tierOf(node);
      return this.modelsByTier[tier] || MODEL_FALLBACKS[tier] || { hidden_dims: [], md5: 'N/A', param_count: null };
    },

    showDetails(node) {
      this.selected = node;
    },

    async connectNew() {
      this.busy = true;
      try {
        const resp = await fetch('/api/connect', { method: 'POST' });
        if (!resp.ok) {
          alert(`Echec de la connexion : ${resp.statusText}`);
        } else {
          await this.loadNodes();
        }
      } catch (err) {
        alert(`Erreur : ${err.message}`);
      } finally {
        this.busy = false;
      }
    },

    isConnected(node) {
      return ['connected', 'registered', 'detecting'].includes(node.status);
    },

    formatCapacity(node) {
      const cpu = node.cpu_cores ?? 'N/A';
      const ram = node.ram_mb ? `${Math.round(node.ram_mb / 1024)} GB` : 'N/A';
      return `${cpu}c / ${ram}`;
    },

    formatCount: formatCountValue,
    formatMs: formatMsValue,
    formatPercent: formatPercentValue,

    formatDims(value) {
      if (!Array.isArray(value) || value.length === 0) return 'N/A';
      return `[${value.join(', ')}]`;
    },

    async initSparkline(nodeId) {
      await this.$nextTick();
      const canvas = document.getElementById(`spark-${nodeId}`);
      if (!canvas || this.sparklines[nodeId]) return;
      this.sparklines[nodeId] = {
        canvas,
        ctx: canvas.getContext('2d'),
        data: [],
      };
      await this.updateSparkline(nodeId);
    },

    async updateSparkline(nodeId) {
      const sparkline = this.sparklines[nodeId];
      if (!sparkline) return;
      try {
        const data = await promQuery(`sum(rate(ids_predictions_total{node_id="${nodeId}"}[30s]))`);
        const value = readPromValue(data) || 0;
        sparkline.data.push(value);
        if (sparkline.data.length > 30) sparkline.data.shift();
        this.drawSparkline(sparkline);
      } catch (_err) {
        // Metrics can be absent during startup. Keep the card stable.
      }
    },

    drawSparkline(sparkline) {
      const { ctx, canvas, data } = sparkline;
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);
      if (data.length < 2) return;

      const max = Math.max(...data, 0.001);
      ctx.strokeStyle = '#31d7c6';
      ctx.lineWidth = 2;
      ctx.beginPath();
      data.forEach((value, index) => {
        const x = (index / (data.length - 1)) * width;
        const y = height - (value / max) * (height - 6) - 3;
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    },
  };
}

function flTab() {
  return {
    health: {},
    schedule: { schedule: {}, triggers: [] },
    runs: [],
    charts: {},
    busy: false,
    interval: null,

    async init() {
      await this.loadAll();
      this.renderCharts();
      this.interval = setInterval(() => this.loadAll(), 10000);
    },

    async loadAll() {
      await Promise.all([
        this.loadHealth(),
        this.loadSchedule(),
        this.loadRuns(),
      ]);
    },

    async loadHealth() {
      try {
        const response = await fetch('/api/fl/health');
        this.health = response.ok ? await response.json() : {};
      } catch (err) {
        console.error('loadHealth failed:', err);
      }
    },

    async loadSchedule() {
      try {
        const response = await fetch('/api/fl/schedule');
        this.schedule = response.ok ? await response.json() : { schedule: {}, triggers: [] };
      } catch (err) {
        console.error('loadSchedule failed:', err);
      }
    },

    async loadRuns() {
      try {
        const response = await fetch('/api/fl/runs?max_results=20');
        const data = response.ok ? await response.json() : { runs: [] };
        this.runs = data.runs || [];
        this.updateCharts();
      } catch (err) {
        console.error('loadRuns failed:', err);
      }
    },

    async triggerTraining() {
      this.busy = true;
      try {
        const response = await fetch('/api/fl/trigger', { method: 'POST' });
        if (!response.ok) {
          alert(`Echec : ${response.statusText}`);
        } else {
          await this.loadAll();
        }
      } catch (err) {
        alert(`Erreur : ${err.message}`);
      } finally {
        this.busy = false;
      }
    },

    getMetric(run, name) {
      if (!run.data || !run.data.metrics) return null;
      const metrics = run.data.metrics;
      if (Array.isArray(metrics)) {
        const metric = metrics.find((item) => item.key === name);
        return metric ? metric.value : null;
      }
      return metrics[name] ?? null;
    },

    formatFloat(value) {
      return formatFloatValue(value, 3);
    },

    formatDate: formatDateValue,

    formatDuration(start, end) {
      if (!start || !end) return 'N/A';
      const startedAt = typeof start === 'number' ? start : new Date(start).getTime();
      const endedAt = typeof end === 'number' ? end : new Date(end).getTime();
      const diff = Math.max(0, (endedAt - startedAt) / 1000);
      if (diff < 60) return `${Math.round(diff)}s`;
      if (diff < 3600) return `${Math.round(diff / 60)}min`;
      return `${(diff / 3600).toFixed(1)}h`;
    },

    renderCharts() {
      if (typeof Chart === 'undefined') return;
      const baseConfig = {
        type: 'line',
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { color: '#1e2d45' }, ticks: { color: '#7f93b4', maxTicksLimit: 6 } },
            y: { grid: { color: '#1e2d45' }, ticks: { color: '#7f93b4', precision: 3 } },
          },
          elements: { point: { radius: 2 } },
        },
      };

      const charts = [
        { id: 'chart-f1', metric: 'fit_f1_macro', color: '#31d7c6' },
        { id: 'chart-acc', metric: 'fit_accuracy', color: '#64d486' },
        { id: 'chart-benign', metric: 'fit_recall_macro', color: '#c99cff' },
        { id: 'chart-loss', metric: 'global_loss', color: '#ff6b7a' },
      ];

      for (const item of charts) {
        const canvas = document.getElementById(item.id);
        if (!canvas || this.charts[item.id]) continue;
        this.charts[item.id] = new Chart(canvas, {
          ...baseConfig,
          data: {
            labels: [],
            datasets: [{
              data: [],
              borderColor: item.color,
              backgroundColor: `${item.color}22`,
              tension: 0.25,
              borderWidth: 2,
              fill: true,
            }],
          },
        });
        this.charts[item.id].metric = item.metric;
      }
      this.updateCharts();
    },

    updateCharts() {
      if (!this.runs.length) return;
      const sorted = [...this.runs].sort((a, b) => (a.info?.start_time || 0) - (b.info?.start_time || 0));

      Object.values(this.charts).forEach((chart) => {
        const rows = sorted
          .map((run) => ({ run, value: this.getMetric(run, chart.metric) }))
          .filter((row) => row.value !== null && !Number.isNaN(Number(row.value)));

        chart.data.labels = rows.map((_row, index) => `R${index + 1}`);
        chart.data.datasets[0].data = rows.map((row) => Number(row.value));
        chart.update('none');
      });
    },
  };
}

function qiTab() {
  return {
    raw: null,
    methodsList: [],
    metricsOrder: [],
    radarChart: null,

    async init() {
      await this.loadAll();
      this.renderRadar();
    },

    async loadAll() {
      try {
        const response = await fetch('/api/qi/overview');
        if (!response.ok) throw new Error(`overview ${response.status}`);
        this.raw = await response.json();

        const methodsObj = this.raw.methods || {};
        this.methodsList = Object.keys(methodsObj).map((key) => ({
          id: key,
          ...methodsObj[key],
          sources: methodsObj[key].sources || [],
        }));
        this.metricsOrder = this.raw.metrics_order || [];
      } catch (err) {
        console.error('qiTab.loadAll failed:', err);
        alert(`Echec chargement metriques QI : ${err.message}`);
      }
    },

    formatValue(methodId, metricId) {
      const method = (this.raw?.methods || {})[methodId];
      if (!method?.metrics) return 'N/A';
      const value = method.metrics[metricId];
      if (value === null || value === undefined) return 'N/A';

      if (metricId === 'false_positive_rate') return Number(value).toFixed(4);
      if (metricId === 'latency_ms') return `${Number(value).toFixed(1)} ms`;
      if (metricId === 'bandwidth_mb_round') return `${Number(value).toFixed(1)} MB`;
      if (metricId === 'memory_mb') return `${Number(value).toFixed(3)} MB`;
      if (typeof value === 'number') return value.toFixed(3);
      return String(value);
    },

    statusBadgeClass: statusClass,

    statusLabel(status) {
      if (status === 'measured') return 'MEASURED';
      if (status === 'expected') return 'EXPECTED';
      if (status === 'literature') return 'LITERATURE';
      if (status === 'planned') return 'PLANNED';
      return 'N/A';
    },

    getCellClass(methodId, metric) {
      const values = this.methodsList
        .map((method) => ({
          id: method.id,
          value: this.raw.methods[method.id]?.metrics?.[metric.id],
        }))
        .filter((item) => item.value !== null && item.value !== undefined);

      if (values.length === 0) return '';

      const sorted = [...values].sort((a, b) => (
        metric.higher_is_better ? b.value - a.value : a.value - b.value
      ));
      const best = sorted[0]?.id;
      const baselineId = 'classical_baseline';

      if (methodId === best && best !== baselineId) return 'qi-cell-best';
      if (methodId === baselineId) return 'qi-cell-baseline';
      return '';
    },

    renderRadar() {
      const canvas = document.getElementById('qi-radar-chart');
      if (!canvas || !this.raw || typeof Chart === 'undefined') return;

      const radarMetrics = (this.metricsOrder || []).filter((metric) => metric.radar);
      const labels = radarMetrics.map((metric) => metric.label);

      const minMax = {};
      radarMetrics.forEach((metric) => {
        const values = this.methodsList
          .map((method) => this.raw.methods[method.id]?.metrics?.[metric.id])
          .filter((value) => value !== null && value !== undefined);
        minMax[metric.id] = { min: Math.min(...values), max: Math.max(...values) };
      });

      const datasets = this.methodsList.map((method) => {
        const data = radarMetrics.map((metric) => {
          const value = this.raw.methods[method.id]?.metrics?.[metric.id];
          if (value === null || value === undefined) return 0;
          const { min, max } = minMax[metric.id];
          if (max === min) return 0.5;
          let normalized = (value - min) / (max - min);
          if (!metric.higher_is_better) normalized = 1 - normalized;
          return Math.max(0, Math.min(1, normalized));
        });
        return {
          label: method.label,
          data,
          borderColor: method.color,
          backgroundColor: `${method.color}24`,
          borderWidth: 2,
          pointRadius: 3,
        };
      });

      this.radarChart = new Chart(canvas, {
        type: 'radar',
        data: { labels, datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            r: {
              beginAtZero: true,
              min: 0,
              max: 1,
              grid: { color: '#1e2d45' },
              angleLines: { color: '#1e2d45' },
              ticks: { display: false, stepSize: 0.2 },
              pointLabels: { color: '#9fb2d1', font: { size: 11 } },
            },
          },
        },
      });
    },
  };
}

function monitoringTab() {
  return {
    services: [],
    targets: [],
    interval: null,

    async init() {
      await this.refresh();
      this.interval = setInterval(() => this.refresh(), 10000);
    },

    async refresh() {
      await Promise.all([this.loadServices(), this.loadTargets()]);
    },

    async loadServices() {
      try {
        const response = await fetch('/api/system/health');
        const data = response.ok ? await response.json() : { services: [] };
        this.services = data.services || [];
      } catch (err) {
        console.error('loadServices failed:', err);
        this.services = [];
      }
    },

    async loadTargets() {
      try {
        const [upPayload, durationPayload] = await Promise.all([
          promQuery('up'),
          promQuery('scrape_duration_seconds'),
        ]);
        const durations = {};
        readPromVector(durationPayload).forEach((row) => {
          durations[`${row.metric.job}|${row.metric.instance}`] = row.value;
        });
        this.targets = readPromVector(upPayload)
          .map((row) => {
            const job = row.metric.job || 'unknown';
            const instance = row.metric.instance || 'unknown';
            return {
              job,
              instance,
              up: row.value >= 1,
              extended: ['iot-node-4', 'iot-node-5'].includes(job),
              scrapeDuration: durations[`${job}|${instance}`] ?? null,
            };
          })
          .sort((a, b) => a.job.localeCompare(b.job));
      } catch (err) {
        console.error('loadTargets failed:', err);
        this.targets = [];
      }
    },

    formatMs: formatMsValue,
  };
}

function systemStatus() {
  return headerStatus();
}

function scenarioSelector() {
  return {
    scenarios: [],
    selected: 'live',
    playing: false,
    currentScenario: null,
    timers: [],

    async init() {
      await this.loadScenarios();
      this.broadcastMode('live');
    },

    async loadScenarios() {
      try {
        const response = await fetch('/api/scenarios');
        const data = await response.json();
        this.scenarios = data.scenarios || [];
      } catch (err) {
        console.error('loadScenarios failed:', err);
      }
    },

    async onChange() {
      this.stop();
      if (this.selected !== 'live') {
        await this.loadScenario(this.selected);
      } else {
        this.currentScenario = null;
        this.broadcastMode('live');
      }
    },

    async loadScenario(id) {
      try {
        const response = await fetch(`/api/scenarios/${id}`);
        if (!response.ok) throw new Error(`scenario ${response.status}`);
        this.currentScenario = await response.json();
      } catch (err) {
        alert(`Echec chargement scenario: ${err.message}`);
      }
    },

    play() {
      if (!this.currentScenario) return;
      this.stop();
      this.playing = true;
      this.broadcastMode('replay', this.currentScenario);

      (this.currentScenario.events || []).forEach((event) => {
        const timer = setTimeout(() => {
          window.dispatchEvent(new CustomEvent('scenario-event', { detail: event }));
        }, event.t * 1000);
        this.timers.push(timer);
      });

      const endTimer = setTimeout(() => {
        this.stop();
        this.selected = 'live';
        this.currentScenario = null;
        this.broadcastMode('live');
      }, ((this.currentScenario.duration_seconds || 0) + 2) * 1000);
      this.timers.push(endTimer);
    },

    stop() {
      this.timers.forEach(clearTimeout);
      this.timers = [];
      this.playing = false;
    },

    broadcastMode(mode, scenario = null) {
      window.dispatchEvent(new CustomEvent('mode-change', {
        detail: { mode, scenario },
      }));
    },
  };
}
