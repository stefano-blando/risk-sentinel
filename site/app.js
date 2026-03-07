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

  if (stress <= 0.5) {
    return mixColor(cool, warm, stress / 0.5);
  }
  return mixColor(warm, hot, (stress - 0.5) / 0.5);
}

function startNetworkDemo() {
  const canvas = document.getElementById("network-canvas");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let nodes = [];
  let edges = [];
  let centerX = 0;
  let centerY = 0;
  let maxRadius = 1;

  function buildGraph(width, height) {
    const rand = mulberry32(42);
    centerX = width * 0.5;
    centerY = height * 0.53;
    maxRadius = Math.min(width, height) * 0.46;

    const count = width < 760 ? 58 : 86;
    nodes = [{ x: centerX, y: centerY, size: 4.6, phase: 0 }];

    for (let i = 1; i < count; i += 1) {
      const t = i / (count - 1);
      const angle = i * 2.399963 + rand() * 0.45;
      const radius = 22 + Math.pow(t, 0.86) * maxRadius;
      const jitter = (rand() - 0.5) * 28;
      nodes.push({
        x: centerX + Math.cos(angle) * radius + jitter,
        y: centerY + Math.sin(angle) * radius + jitter,
        size: 1.8 + rand() * 2.5,
        phase: rand() * Math.PI * 2
      });
    }

    const edgeSet = new Set();
    edges = [];

    for (let i = 1; i < nodes.length; i += 1) {
      const nearest = [];
      for (let j = 1; j < nodes.length; j += 1) {
        if (i === j) continue;
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
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

  function nodeStress(node, t) {
    const dx = node.x - centerX;
    const dy = node.y - centerY;
    const dist = Math.hypot(dx, dy);
    const waveA = (t * 0.13) % maxRadius;
    const waveB = (waveA + maxRadius * 0.52) % maxRadius;
    const band = 56;

    const influenceA = Math.max(0, 1 - Math.abs(dist - waveA) / band);
    const influenceB = Math.max(0, 1 - Math.abs(dist - waveB) / band) * 0.58;
    const centerBoost = dist < 9 ? 1 : 0;

    return Math.min(1, centerBoost + Math.max(influenceA, influenceB) * 0.95);
  }

  function draw(nowMs) {
    const now = nowMs || performance.now();
    const t = now;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    ctx.clearRect(0, 0, width, height);

    const grad = ctx.createLinearGradient(0, 0, width, height);
    grad.addColorStop(0, "rgba(7,11,20,0.92)");
    grad.addColorStop(1, "rgba(13,23,42,0.88)");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, width, height);

    const ringRadius = (t * 0.13) % maxRadius;
    [ringRadius, (ringRadius + maxRadius * 0.52) % maxRadius].forEach((r, idx) => {
      ctx.beginPath();
      ctx.arc(centerX, centerY, r, 0, Math.PI * 2);
      ctx.lineWidth = idx === 0 ? 1.3 : 1.0;
      ctx.strokeStyle = idx === 0 ? "rgba(251,146,60,0.34)" : "rgba(56,189,248,0.2)";
      ctx.stroke();
    });

    edges.forEach(([aIdx, bIdx]) => {
      const a = nodes[aIdx];
      const b = nodes[bIdx];
      const edgeStress = (nodeStress(a, t) + nodeStress(b, t)) * 0.5;
      const [r, g, bCol] = stressColor(edgeStress);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = `rgba(${r}, ${g}, ${bCol}, ${0.08 + edgeStress * 0.36})`;
      ctx.lineWidth = 0.6 + edgeStress * 0.85;
      ctx.stroke();
    });

    nodes.forEach((node, idx) => {
      const stress = nodeStress(node, t);
      const [r, g, bCol] = stressColor(stress);
      const pulse = prefersReducedMotion ? 0 : Math.sin(t * 0.003 + node.phase) * 0.35;
      const radius = node.size + stress * 2.0 + pulse;

      ctx.beginPath();
      ctx.arc(node.x, node.y, Math.max(1.25, radius), 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${r}, ${g}, ${bCol}, 0.9)`;
      ctx.shadowBlur = 12 + stress * 14;
      ctx.shadowColor = `rgba(${r}, ${g}, ${bCol}, 0.72)`;
      ctx.fill();
      ctx.shadowBlur = 0;

      if (idx === 0) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, 6.2 + (prefersReducedMotion ? 0 : Math.sin(t * 0.008) * 1.1), 0, Math.PI * 2);
        ctx.fillStyle = "rgba(239,68,68,0.92)";
        ctx.fill();
      }
    });
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

  resize();
  window.addEventListener("resize", resize);

  if (!prefersReducedMotion) {
    const loop = now => {
      draw(now);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }
}

startNetworkDemo();
