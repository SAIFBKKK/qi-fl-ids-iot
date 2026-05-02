function iotTab() {
  return {
    nodes: [],
    selected: null,
    busy: false,
    error: null,
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
      } catch (err) {
        this.error = `Dashboard indisponible: ${err.message}`;
        console.error('loadNodes failed:', err);
      }
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
    }
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
