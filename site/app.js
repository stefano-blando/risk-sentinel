const observer = new IntersectionObserver(
  entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
      }
    });
  },
  { threshold: 0.18 }
);

document.querySelectorAll(".reveal").forEach(el => observer.observe(el));

document.querySelectorAll(".metric[data-count]").forEach(node => {
  const target = Number(node.dataset.count);
  if (!Number.isFinite(target)) return;

  const duration = 900;
  const start = performance.now();

  const tick = now => {
    const p = Math.min((now - start) / duration, 1);
    const value = Math.floor(target * p).toLocaleString("en-US");
    node.textContent = value;
    if (p < 1) requestAnimationFrame(tick);
  };

  requestAnimationFrame(tick);
});

const statusChip = document.getElementById("status-chip");
if (statusChip) {
  const states = [
    "Control plane ready",
    "Evidence gate active",
    "Critic validation online"
  ];
  let idx = 0;
  window.setInterval(() => {
    idx = (idx + 1) % states.length;
    statusChip.textContent = states[idx];
  }, 2400);
}

function mulberry32(seed) {
  let t = seed;
  return function rand() {
    t += 0x6D2B79F5;
    let x = Math.imul(t ^ (t >>> 15), t | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function mixColor(a, b, k) {
  const t = Math.max(0, Math.min(1, k));
  const r = Math.round(a[0] + (b[0] - a[0]) * t);
  const g = Math.round(a[1] + (b[1] - a[1]) * t);
  const bl = Math.round(a[2] + (b[2] - a[2]) * t);
  return [r, g, bl];
}

function stressColor(stress) {
  const cool = [56, 189, 248];
  const warm = [251, 146, 60];
  const hot = [239, 68, 68];

  if (stress <= 0.55) {
    return mixColor(cool, warm, stress / 0.55);
  }
  return mixColor(warm, hot, (stress - 0.55) / 0.45);
}

function influence(distance, waveRadius, band) {
  return Math.max(0, 1 - Math.abs(distance - waveRadius) / band);
}

function startNetworkDemo() {
  const canvas = document.getElementById("network-canvas");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const modeButtons = Array.from(document.querySelectorAll("[data-net-mode]"));
  const toggleBtn = document.getElementById("net-toggle");
  const modeLabelEl = document.getElementById("network-mode-label");
  const waveEl = document.getElementById("network-wave-depth");
  const affectedEl = document.getElementById("network-affected");

  const modeConfig = {
    calm: {
      label: "Calm",
      speed: 0.075,
      secondary: false,
      tertiary: false,
      threshold: 0.54,
      edgeAlpha: 0.16,
      pulse: 0.2,
      baseline: 0.05
    },
    stress: {
      label: "Stress",
      speed: 0.135,
      secondary: true,
      tertiary: false,
      threshold: 0.36,
      edgeAlpha: 0.34,
      pulse: 0.44,
      baseline: 0.1
    },
    crisis: {
      label: "Crisis",
      speed: 0.205,
      secondary: true,
      tertiary: true,
      threshold: 0.24,
      edgeAlpha: 0.5,
      pulse: 0.66,
      baseline: 0.14
    }
  };

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let currentMode = "stress";
  let running = !prefersReducedMotion;
  let centerX = 0;
  let centerY = 0;
  let maxRadius = 1;
  let nodes = [];
  let edges = [];
  let secondaryCenter = { x: 0, y: 0 };
  let tertiaryCenter = { x: 0, y: 0 };
  let lastStatsAt = 0;

  function setMode(mode) {
    if (!modeConfig[mode]) return;
    currentMode = mode;
    modeButtons.forEach(btn => {
      btn.classList.toggle("is-active", btn.dataset.netMode === mode);
    });
    if (modeLabelEl) modeLabelEl.textContent = modeConfig[mode].label;
  }

  function buildGraph(width, height) {
    const rand = mulberry32(42);
    centerX = width * 0.5;
    centerY = height * 0.54;
    maxRadius = Math.min(width, height) * 0.44;

    const count = width < 760 ? 56 : 90;
    nodes = [{ x0: centerX, y0: centerY, x: centerX, y: centerY, size: 4.8, phase: 0, wobble: 0.5 }];

    for (let i = 1; i < count; i += 1) {
      const t = i / (count - 1);
      const angle = i * 2.399963 + rand() * 0.45;
      const radius = 24 + Math.pow(t, 0.86) * maxRadius;
      const jitter = (rand() - 0.5) * 22;
      const x0 = centerX + Math.cos(angle) * radius + jitter;
      const y0 = centerY + Math.sin(angle) * radius + jitter;
      nodes.push({
        x0,
        y0,
        x: x0,
        y: y0,
        size: 1.8 + rand() * 2.4,
        phase: rand() * Math.PI * 2,
        wobble: 0.35 + rand() * 1.05
      });
    }

    secondaryCenter = {
      x: centerX - maxRadius * 0.38,
      y: centerY + maxRadius * 0.18
    };
    tertiaryCenter = {
      x: centerX + maxRadius * 0.31,
      y: centerY - maxRadius * 0.22
    };

    const edgeSet = new Set();
    edges = [];

    for (let i = 1; i < nodes.length; i += 1) {
      const nearest = [];
      for (let j = 1; j < nodes.length; j += 1) {
        if (i === j) continue;
        const dx = nodes[i].x0 - nodes[j].x0;
        const dy = nodes[i].y0 - nodes[j].y0;
        nearest.push({ j, d2: dx * dx + dy * dy });
      }
      nearest.sort((a, b) => a.d2 - b.d2);

      for (let k = 0; k < 3; k += 1) {
        const j = nearest[k].j;
        const a = Math.min(i, j);
        const b = Math.max(i, j);
        const key = `${a}-${b}`;
        if (!edgeSet.has(key)) {
          edgeSet.add(key);
          edges.push([a, b]);
        }
      }

      if (i % 4 === 0) {
        const key = `0-${i}`;
        if (!edgeSet.has(key)) {
          edgeSet.add(key);
          edges.push([0, i]);
        }
      }
    }
  }

  function updateNodePositions(t) {
    nodes.forEach((node, idx) => {
      if (idx === 0) {
        node.x = centerX;
        node.y = centerY;
        return;
      }
      const drift = prefersReducedMotion ? 0 : node.wobble;
      node.x = node.x0 + Math.sin(t * 0.00052 + node.phase) * drift * 8.5;
      node.y = node.y0 + Math.cos(t * 0.00047 + node.phase) * drift * 8.5;
    });
  }

  function nodeStress(node, t, cfg) {
    const ringA = (t * cfg.speed) % maxRadius;
    const ringB = (ringA + maxRadius * 0.53) % maxRadius;

    const d0 = Math.hypot(node.x - centerX, node.y - centerY);
    const inf0 = Math.max(influence(d0, ringA, 58), influence(d0, ringB, 58) * 0.58);

    let inf1 = 0;
    if (cfg.secondary) {
      const d1 = Math.hypot(node.x - secondaryCenter.x, node.y - secondaryCenter.y);
      const ringS = (ringA * 0.82 + maxRadius * 0.2) % maxRadius;
      inf1 = influence(d1, ringS, 62) * 0.54;
    }

    let inf2 = 0;
    if (cfg.tertiary) {
      const d2 = Math.hypot(node.x - tertiaryCenter.x, node.y - tertiaryCenter.y);
      const ringT = (ringA * 0.67 + maxRadius * 0.38) % maxRadius;
      inf2 = influence(d2, ringT, 66) * 0.4;
    }

    const centerBoost = d0 < 9 ? 1 : 0;
    return Math.min(1, Math.max(centerBoost, inf0, inf1, inf2) + cfg.baseline);
  }

  function draw(nowMs) {
    const now = nowMs || performance.now();
    const cfg = modeConfig[currentMode];
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    updateNodePositions(now);

    ctx.clearRect(0, 0, width, height);

    const grad = ctx.createLinearGradient(0, 0, width, height);
    grad.addColorStop(0, "rgba(7,11,20,0.92)");
    grad.addColorStop(1, "rgba(13,23,42,0.88)");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, width, height);

    const ringMain = (now * cfg.speed) % maxRadius;
    const ringSecond = (ringMain + maxRadius * 0.53) % maxRadius;

    [ringMain, ringSecond].forEach((r, idx) => {
      ctx.beginPath();
      ctx.arc(centerX, centerY, r, 0, Math.PI * 2);
      ctx.lineWidth = idx === 0 ? 1.35 : 1.0;
      ctx.strokeStyle = idx === 0 ? "rgba(251,146,60,0.36)" : "rgba(56,189,248,0.2)";
      ctx.stroke();
    });

    if (cfg.secondary) {
      ctx.beginPath();
      ctx.arc(secondaryCenter.x, secondaryCenter.y, (ringMain * 0.82 + maxRadius * 0.2) % maxRadius, 0, Math.PI * 2);
      ctx.lineWidth = 0.85;
      ctx.strokeStyle = "rgba(96,165,250,0.22)";
      ctx.stroke();
    }

    if (cfg.tertiary) {
      ctx.beginPath();
      ctx.arc(tertiaryCenter.x, tertiaryCenter.y, (ringMain * 0.67 + maxRadius * 0.38) % maxRadius, 0, Math.PI * 2);
      ctx.lineWidth = 0.8;
      ctx.strokeStyle = "rgba(239,68,68,0.2)";
      ctx.stroke();
    }

    edges.forEach(([aIdx, bIdx]) => {
      const a = nodes[aIdx];
      const b = nodes[bIdx];
      const edgeStress = (nodeStress(a, now, cfg) + nodeStress(b, now, cfg)) * 0.5;
      const [r, g, bCol] = stressColor(edgeStress);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = `rgba(${r}, ${g}, ${bCol}, ${0.06 + edgeStress * cfg.edgeAlpha})`;
      ctx.lineWidth = 0.6 + edgeStress * 0.92;
      ctx.stroke();
    });

    let affected = 0;
    nodes.forEach((node, idx) => {
      const stress = nodeStress(node, now, cfg);
      if (stress >= cfg.threshold) affected += 1;

      const [r, g, bCol] = stressColor(stress);
      const pulse = prefersReducedMotion ? 0 : Math.sin(now * 0.003 + node.phase) * cfg.pulse;
      const radius = node.size + stress * 2.1 + pulse;

      ctx.beginPath();
      ctx.arc(node.x, node.y, Math.max(1.2, radius), 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${r}, ${g}, ${bCol}, 0.9)`;
      ctx.shadowBlur = 10 + stress * 14;
      ctx.shadowColor = `rgba(${r}, ${g}, ${bCol}, 0.72)`;
      ctx.fill();
      ctx.shadowBlur = 0;

      if (idx === 0) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, 6.3 + (prefersReducedMotion ? 0 : Math.sin(now * 0.008) * 1.2), 0, Math.PI * 2);
        ctx.fillStyle = "rgba(239,68,68,0.95)";
        ctx.fill();
      }
    });

    if (now - lastStatsAt > 120) {
      const waveDepth = Math.floor((ringMain / maxRadius) * 12) + 1;
      const affectedPct = Math.round((affected / nodes.length) * 100);
      if (waveEl) waveEl.textContent = String(waveDepth);
      if (affectedEl) affectedEl.textContent = `${affectedPct}%`;
      lastStatsAt = now;
    }
  }

  function resize() {
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    buildGraph(width, height);
    draw(performance.now());
  }

  modeButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      setMode(btn.dataset.netMode);
      draw(performance.now());
    });
  });

  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      running = !running;
      toggleBtn.textContent = running ? "Pause" : "Play";
      if (running && !prefersReducedMotion) {
        requestAnimationFrame(loop);
      }
    });
    if (prefersReducedMotion) {
      toggleBtn.textContent = "Play";
    }
  }

  function loop(now) {
    if (!running || prefersReducedMotion) return;
    draw(now);
    requestAnimationFrame(loop);
  }

  setMode(currentMode);
  resize();
  window.addEventListener("resize", resize);

  if (!prefersReducedMotion) {
    requestAnimationFrame(loop);
  }
}

startNetworkDemo();
