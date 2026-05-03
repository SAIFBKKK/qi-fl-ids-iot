function iotTab() {
  return {
    nodes: [],
    selected: null,
    busy: false,
    error: null,
    filterStatus: 'all',
    filterTier: 'all',
    sparklines: {},
    async loadNodes() {
      try {
        const resp = await fetch('/api/nodes');
        if (!resp.ok) throw new Error(`nodes ${resp.status}`);
        const data = await resp.json();
        const nodes = data.nodes || [];

        const modelsResp = await fetch('/api/models');
        if (!modelsResp.ok) throw new Error(`models ${modelsResp.status}`);
        const modelsData = await modelsResp.json();
        const modelRows = modelsData.tiers || modelsData.models || [];
        const md5ByTier = {};
        modelRows.forEach((tier) => {
          if (tier.md5) md5ByTier[tier.tier] = tier.md5;
        });

        nodes.forEach((node) => {
          node.assigned_model_md5 = md5ByTier[node.assigned_tier] || null;
        });
        this.nodes = nodes;
        this.error = null;
        setTimeout(() => {
          this.nodes.forEach((node) => this.updateSparkline(node.node_id));
        }, 0);
      } catch (err) {
        this.error = `Dashboard indisponible: ${err.message}`;
        console.error('loadNodes failed:', err);
      }
    },
    get filteredNodes() {
      return this.nodes.filter((node) => {
        const statusOk = this.filterStatus === 'all' || node.status === this.filterStatus;
        const tierOk = this.filterTier === 'all' || node.assigned_tier === this.filterTier;
        return statusOk && tierOk;
      });
    },
    showDetails(node) {
      this.selected = node;
    },
    async connectNew() {
      this.busy = true;
      try {
        const resp = await fetch('/api/connect', { method: 'POST' });
        if (!resp.ok) {
          alert(`Échec de la connexion : ${resp.statusText}`);
        } else {
          await this.loadNodes();
        }
      } catch (err) {
        alert(`Erreur : ${err.message}`);
      } finally {
        this.busy = false;
      }
    },
    formatCapacity(node) {
      const cpu = node.cpu_cores ?? '-';
      const ram = node.ram_mb ? `${Math.round(node.ram_mb / 1024)}GB` : '-';
      return `${cpu}c / ${ram}`;
    },
    formatLatency(node) {
      return node.avg_latency_ms ? `${Number(node.avg_latency_ms).toFixed(1)}ms` : '-';
    },
    isConnected(node) {
      return ['connected', 'registered', 'detecting'].includes(node.status);
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
        const query = `sum(rate(ids_predictions_total{node_id="${nodeId}"}[30s]))`;
        const response = await fetch(`/api/prometheus/query?q=${encodeURIComponent(query)}`);
        const data = await response.json();
        const value = parseFloat(data?.data?.result?.[0]?.value?.[1] || '0');
        sparkline.data.push(Number.isNaN(value) ? 0 : value);
        if (sparkline.data.length > 30) sparkline.data.shift();
        this.drawSparkline(sparkline);
      } catch (_err) {
        // Prometheus metrics may be absent during startup or replay demos.
      }
    },
    drawSparkline(sparkline) {
      const { ctx, canvas, data } = sparkline;
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);
      if (data.length < 2) return;

      const max = Math.max(...data, 0.001);
      ctx.strokeStyle = '#0F6E56';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      data.forEach((value, index) => {
        const x = (index / (data.length - 1)) * width;
        const y = height - (value / max) * (height - 4) - 2;
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
          alert(`Échec : ${response.statusText}`);
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
      if (value === null || value === undefined || isNaN(value)) return '-';
      return Number(value).toFixed(3);
    },

    formatDate(value) {
      if (!value) return null;
      const date = typeof value === 'number' ? new Date(value) : new Date(value);
      if (isNaN(date.getTime())) return null;
      return date.toLocaleString('fr-FR', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    },

    formatDuration(start, end) {
      if (!start || !end) return '-';
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
            x: { grid: { display: false }, ticks: { maxTicksLimit: 6 } },
            y: { beginAtZero: false, ticks: { precision: 3 } },
          },
          elements: { point: { radius: 2 } },
        },
      };

      const charts = [
        { id: 'chart-f1', metric: 'fit_f1_macro', color: '#185FA5' },
        { id: 'chart-acc', metric: 'fit_accuracy', color: '#0F6E56' },
        { id: 'chart-benign', metric: 'fit_recall_macro', color: '#854F0B' },
        { id: 'chart-loss', metric: 'global_loss', color: '#A32D2D' },
      ];

      for (const item of charts) {
        const canvas = document.getElementById(item.id);
        if (!canvas) continue;
        this.charts[item.id] = new Chart(canvas, {
          ...baseConfig,
          data: {
            labels: [],
            datasets: [{
              data: [],
              borderColor: item.color,
              backgroundColor: `${item.color}20`,
              tension: 0.2,
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
          .filter((row) => row.value !== null && !isNaN(row.value));

        chart.data.labels = rows.map((_row, index) => `R${index + 1}`);
        chart.data.datasets[0].data = rows.map((row) => row.value);
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
      if (!method?.metrics) return '-';
      const value = method.metrics[metricId];
      if (value === null || value === undefined) return '-';

      if (metricId === 'false_positive_rate') return value.toFixed(4);
      if (metricId === 'latency_ms') return value.toFixed(1);
      if (metricId === 'bandwidth_mb_round') return value.toFixed(1);
      if (metricId === 'memory_mb') return value.toFixed(3);
      if (typeof value === 'number') return value.toFixed(3);
      return String(value);
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
          backgroundColor: `${method.color}20`,
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
          scales: {
            r: {
              beginAtZero: true,
              min: 0,
              max: 1,
              ticks: { display: false, stepSize: 0.2 },
              pointLabels: { font: { size: 11 } },
            },
          },
          plugins: {
            legend: { display: false },
          },
        },
      });
    },
  };
}

function systemStatus() {
  return {
    overall: 'unknown',
    label: 'Verification...',
    title: '',

    async init() {
      await this.refresh();
      setInterval(() => this.refresh(), 10000);
    },

    async refresh() {
      try {
        const response = await fetch('/api/system/health');
        const data = await response.json();
        this.overall = data.overall;
        this.label = `${data.ups}/${data.total} services`;
        this.title = (data.services || []).map((service) => (
          `${service.service}: ${service.status}${service.latency_ms ? ` (${service.latency_ms}ms)` : ''}`
        )).join('\n');
      } catch (_err) {
        this.overall = 'down';
        this.label = 'dashboard isole';
        this.title = 'Erreur fetch /api/system/health';
      }
    },
  };
}
