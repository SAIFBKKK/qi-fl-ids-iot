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
