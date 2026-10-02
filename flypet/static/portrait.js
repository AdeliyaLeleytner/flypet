// Detailed side-view fly "portrait" with behaviour-driven animation. Builds an SVG into #portrait.
(function () {
  const NS = 'http://www.w3.org/2000/svg';
  const el = (name, attrs = {}, parent) => { const e = document.createElementNS(NS, name); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
  const rnd = (seed => () => (seed = (seed * 16807) % 2147483647) / 2147483647)(42);
  const host = document.getElementById('portrait'); if (!host) return;
  const svg = el('svg', { viewBox: '0 0 900 600', xmlns: NS, id: 'portraitsvg' }, host);
  const defs = el('defs', {}, svg);
  const grad = (id, stops, type = 'radialGradient', extra = {}) => { const g = el(type, { id, ...extra }, defs); stops.forEach(([o, c, op]) => el('stop', { offset: o, 'stop-color': c, 'stop-opacity': op ?? 1 }, g)); return g; };
  grad('bg', [[0, '#3a2a22'], [1, '#12100e']], 'radialGradient', { cx: '35%', cy: '40%', r: '80%' });
  grad('cuticle', [[0, '#e0b26a'], [0.55, '#a9732f'], [1, '#5a3a17']], 'radialGradient', { cx: '35%', cy: '30%', r: '85%' });
  grad('cuticleDark', [[0, '#8c5a26'], [0.6, '#5d3814'], [1, '#2e1a08']], 'radialGradient', { cx: '35%', cy: '30%', r: '85%' });
  grad('eyeBase', [[0, '#ff8f6e'], [0.45, '#c8422b'], [0.8, '#6e1a12'], [1, '#2a0806']], 'radialGradient', { cx: '38%', cy: '35%', r: '75%' });
  grad('eyeGloss', [[0, '#fff', 0.55], [0.3, '#fff', 0.12], [1, '#fff', 0]], 'radialGradient', { cx: '30%', cy: '25%', r: '45%' });
  grad('wingG', [[0, '#e8f2ff', 0.55], [1, '#b8c8dd', 0.25]], 'linearGradient', { x1: 0, y1: 0, x2: 1, y2: 1 });
  grad('labG', [[0, '#f2d9a6'], [1, '#b8843c']], 'radialGradient', { cx: '50%', cy: '40%', r: '70%' });
  // facet pattern: hex-packed dots
  const pat = el('pattern', { id: 'facets', width: 9, height: 15.6, patternUnits: 'userSpaceOnUse' }, defs);
  el('circle', { cx: 4.5, cy: 3.9, r: 3.6, fill: '#000', opacity: 0.22 }, pat); el('circle', { cx: 0, cy: 11.7, r: 3.6, fill: '#000', opacity: 0.22 }, pat); el('circle', { cx: 9, cy: 11.7, r: 3.6, fill: '#000', opacity: 0.22 }, pat);
  el('circle', { cx: 3.6, cy: 3, r: 1.1, fill: '#fff', opacity: 0.18 }, pat); el('circle', { cx: 8.1, cy: 10.8, r: 1.1, fill: '#fff', opacity: 0.18 }, pat);
  const blur = el('filter', { id: 'soft', x: '-20%', y: '-20%', width: '140%', height: '140%' }, defs); el('feGaussianBlur', { stdDeviation: 1.2 }, blur);
  const shadowF = el('filter', { id: 'shadow', x: '-20%', y: '-20%', width: '150%', height: '150%' }, defs); el('feDropShadow', { dx: 4, dy: 8, stdDeviation: 6, 'flood-color': '#000', 'flood-opacity': 0.45 }, shadowF);
  const wingBlur = el('filter', { id: 'wingblur' }, defs); const wb = el('feGaussianBlur', { stdDeviation: 0 }, wingBlur);
  // background + ground
  el('rect', { width: 900, height: 600, fill: 'url(#bg)' }, svg);
  el('ellipse', { cx: 470, cy: 575, rx: 330, ry: 22, fill: '#000', opacity: 0.35, filter: 'url(#soft)' }, svg);
  const body = el('g', { id: 'body', filter: 'url(#shadow)' }, svg);          // everything that bobs
  const bristle = (parent, x, y, angle, len, w = 2, color = '#1a1008') => {
    const a = angle * Math.PI / 180, cx = x + Math.cos(a) * len * 0.5 + (rnd() - 0.5) * 3, cy = y + Math.sin(a) * len * 0.5 - 4;
    el('path', { d: `M${x},${y} Q${cx},${cy} ${x + Math.cos(a) * len},${y + Math.sin(a) * len}`, stroke: color, 'stroke-width': w, fill: 'none', 'stroke-linecap': 'round', opacity: 0.85 }, parent);
  };
  // ---- abdomen
  const abd = el('g', { id: 'abdomen', style: 'transform-origin:640px 340px' }, body);
  el('ellipse', { cx: 760, cy: 345, rx: 165, ry: 100, fill: 'url(#cuticle)' }, abd);
  for (let i = 0; i < 5; i++) { const x = 655 + i * 42; el('path', { d: `M${x},${262 + i * 4} Q${x + 22},${345} ${x + 4},${430 - i * 5}`, stroke: '#3a2210', 'stroke-width': 9 - i, fill: 'none', opacity: 0.55 }, abd); }
  for (let i = 0; i < 60; i++) { const t = rnd() * Math.PI - Math.PI / 2, x = 760 + Math.cos(t) * 165 * (0.5 + rnd() * 0.5), y = 345 + Math.sin(t) * 100 * (0.5 + rnd() * 0.5); bristle(abd, x, y, -60 + rnd() * 30, 10 + rnd() * 14, 1.4); }
  // ---- haltere
  el('path', { d: 'M640,330 Q655,352 668,360', stroke: '#c9a05a', 'stroke-width': 3, fill: 'none' }, body); el('circle', { cx: 670, cy: 362, r: 6, fill: '#e2c47a' }, body);
  // ---- wing (over the abdomen, translucent)
  const wing = el('g', { id: 'wing', transform: 'rotate(-8 600 245)', style: 'transform-origin:600px 245px' }, body);
  el('path', { d: 'M600,245 C650,170 820,140 895,175 C905,215 880,265 820,295 C760,320 660,318 620,290 C605,278 598,262 600,245 Z', fill: 'url(#wingG)', stroke: '#b9c7d8', 'stroke-width': 1.4, filter: 'url(#wingblur)' }, wing);
  el('path', { d: 'M640,200 C700,170 800,160 880,180', stroke: '#fff', 'stroke-width': 6, fill: 'none', opacity: 0.12, filter: 'url(#soft)' }, wing);
  [['M600,245 C700,205 810,180 893,178', 1.6], ['M603,250 C700,232 800,215 895,205', 1.2], ['M606,258 C700,262 790,262 875,240', 1.2], ['M612,270 C690,292 770,300 830,290', 1.1], ['M640,232 L650,300'], ['M700,212 L712,305'], ['M770,192 L785,300'], ['M840,182 L852,275'], ['M760,258 L840,236']].forEach(([d, w]) => el('path', { d, stroke: '#7f91a6', 'stroke-width': w || 1, fill: 'none', opacity: 0.85 }, wing));
  for (let i = 0; i < 40; i++) { const x = 610 + rnd() * 270, y = 180 + rnd() * 110; el('circle', { cx: x, cy: y, r: 0.9, fill: '#5a6a80', opacity: 0.5 }, wing); }
  // ---- legs (far side first, faint)
  const leg = (id, coxa, femurEnd, tibiaEnd, tarsusEnd, faint) => {
    const g = el('g', { id, style: `transform-origin:${coxa[0]}px ${coxa[1]}px`, opacity: faint ? 0.45 : 1 }, body);
    const fem = el('g', { style: `transform-origin:${coxa[0]}px ${coxa[1]}px` }, g);
    el('line', { x1: coxa[0], y1: coxa[1], x2: femurEnd[0], y2: femurEnd[1], stroke: '#b98444', 'stroke-width': 11, 'stroke-linecap': 'round' }, fem);
    const tib = el('g', { style: `transform-origin:${femurEnd[0]}px ${femurEnd[1]}px` }, fem);
    el('line', { x1: femurEnd[0], y1: femurEnd[1], x2: tibiaEnd[0], y2: tibiaEnd[1], stroke: '#a8742f', 'stroke-width': 8, 'stroke-linecap': 'round' }, tib);
    const tar = el('g', { style: `transform-origin:${tibiaEnd[0]}px ${tibiaEnd[1]}px` }, tib);
    for (let s = 0; s < 5; s++) { const t0 = s / 5, t1 = (s + 1) / 5; el('line', { x1: tibiaEnd[0] + (tarsusEnd[0] - tibiaEnd[0]) * t0, y1: tibiaEnd[1] + (tarsusEnd[1] - tibiaEnd[1]) * t0, x2: tibiaEnd[0] + (tarsusEnd[0] - tibiaEnd[0]) * t1, y2: tibiaEnd[1] + (tarsusEnd[1] - tibiaEnd[1]) * t1, stroke: s % 2 ? '#8a5a22' : '#a06a2a', 'stroke-width': 6 - s * 0.6, 'stroke-linecap': 'round' }, tar); }
    el('path', { d: `M${tarsusEnd[0]},${tarsusEnd[1]} l-8,6 M${tarsusEnd[0]},${tarsusEnd[1]} l6,8`, stroke: '#2a1808', 'stroke-width': 2.2, fill: 'none', 'stroke-linecap': 'round' }, tar);
    el('ellipse', { cx: tarsusEnd[0] - 1, cy: tarsusEnd[1] + 4, rx: 6, ry: 3.5, fill: '#e8d8b0', opacity: 0.8 }, tar);
    for (let i = 0; i < 14; i++) { const t = rnd(); const x = coxa[0] + (femurEnd[0] - coxa[0]) * t, y = coxa[1] + (femurEnd[1] - coxa[1]) * t; bristle(fem, x, y, 200 + rnd() * 60, 8 + rnd() * 8, 1.3); }
    for (let i = 0; i < 12; i++) { const t = rnd(); const x = femurEnd[0] + (tibiaEnd[0] - femurEnd[0]) * t, y = femurEnd[1] + (tibiaEnd[1] - femurEnd[1]) * t; bristle(tib, x, y, 190 + rnd() * 60, 6 + rnd() * 7, 1.1); }
    return { g, fem, tib, tar };
  };
  const legFarHind = leg('legFH', [650, 395], [735, 470], [775, 560], [820, 585], true);
  const legFarMid = leg('legFM', [575, 400], [625, 495], [650, 570], [700, 590], true);
  // ---- thorax
  const thorax = el('g', { id: 'thorax', style: 'transform-origin:560px 300px' }, body);
  el('ellipse', { cx: 560, cy: 300, rx: 150, ry: 108, fill: 'url(#cuticle)' }, thorax);
  el('path', { d: 'M430,300 C470,250 640,240 700,290', stroke: '#6b4418', 'stroke-width': 3, fill: 'none', opacity: 0.5 }, thorax);
  el('ellipse', { cx: 540, cy: 250, rx: 70, ry: 22, fill: '#f0cf8a', opacity: 0.25 }, thorax);
  el('path', { d: 'M640,236 C660,260 665,300 655,340', stroke: '#4a2c10', 'stroke-width': 2.5, fill: 'none', opacity: 0.6 }, thorax);
  el('path', { d: 'M430,300 C440,340 470,380 520,400', stroke: '#3b230d', 'stroke-width': 2, fill: 'none', opacity: 0.5 }, thorax);
  for (let i = 0; i < 6; i++) bristle(thorax, 610 + i * 12, 232 + i * 8, -80 + i * 6, 40 - i * 3, 2.6);
  for (let i = 0; i < 90; i++) { const t = rnd() * Math.PI * 2, x = 560 + Math.cos(t) * 150 * (0.35 + rnd() * 0.65), y = 300 + Math.sin(t) * 108 * (0.35 + rnd() * 0.65); if (y > 300) continue; bristle(thorax, x, y, -100 + rnd() * 50, 12 + rnd() * 22, 1.6); }
  for (let i = 0; i < 8; i++) bristle(thorax, 470 + i * 26, 205 + Math.abs(i - 4) * 6, -105 + i * 4, 34, 2.4);
  // ---- near legs
  const legHind = leg('legH', [640, 405], [725, 485], [770, 575], [830, 590], false);
  const legMid = leg('legM', [560, 410], [600, 505], [615, 580], [680, 595], false);
  const legFront = leg('legF', [470, 385], [405, 470], [330, 545], [255, 585], false);
  // ---- head
  const head = el('g', { id: 'head', style: 'transform-origin:420px 300px' }, body);
  el('path', { d: 'M420,250 C420,220 445,205 470,215 L470,360 C450,372 420,360 415,335 Z', fill: '#7b4f1f' }, head);   // neck
  el('circle', { cx: 330, cy: 265, r: 112, fill: 'url(#cuticle)' }, head);
  el('path', { d: 'M300,350 C330,395 372,388 392,350 C380,335 340,332 300,350 Z', fill: 'url(#cuticleDark)' }, head);      // gena / lower face
  el('ellipse', { cx: 395, cy: 235, rx: 22, ry: 70, fill: '#7a2418', opacity: 0.55 }, head);   // far eye edge
  const eye = el('g', { id: 'eye' }, head);
  el('ellipse', { cx: 300, cy: 255, rx: 84, ry: 100, fill: 'url(#eyeBase)' }, eye);
  el('ellipse', { cx: 300, cy: 255, rx: 84, ry: 100, fill: 'url(#facets)' }, eye);
  const gloss = el('ellipse', { cx: 300, cy: 255, rx: 84, ry: 100, fill: 'url(#eyeGloss)' }, eye);
  const glint = el('ellipse', { cx: 268, cy: 205, rx: 18, ry: 9, fill: '#fff', opacity: 0.35, transform: 'rotate(-30 268 205)' }, eye);
  el('ellipse', { cx: 300, cy: 255, rx: 86, ry: 102, fill: 'none', stroke: '#2a0c08', 'stroke-width': 3, opacity: 0.6 }, eye);
  for (let i = 0; i < 70; i++) { const t = rnd() * Math.PI * 2, r = 0.55 + rnd() * 0.42, x = 300 + Math.cos(t) * 84 * r, y = 255 + Math.sin(t) * 100 * r; el('line', { x1: x, y1: y, x2: x + Math.cos(t) * 6, y2: y + Math.sin(t) * 6, stroke: '#1a0806', 'stroke-width': 0.9, opacity: 0.7 }, eye); }
  [[352, 170], [366, 160], [380, 172]].forEach(([x, y]) => el('circle', { cx: x, cy: y, r: 5, fill: '#c9442d', stroke: '#3a0f0a', 'stroke-width': 1.5 }, head)); // ocelli
  for (let i = 0; i < 26; i++) { const t = -2.2 + rnd() * 1.3, x = 330 + Math.cos(t) * 112, y = 265 + Math.sin(t) * 112; bristle(head, x, y, (t * 180 / Math.PI), 16 + rnd() * 22, 2); }
  for (let i = 0; i < 12; i++) bristle(head, 350 + rnd() * 40, 320 + rnd() * 40, 60 + rnd() * 50, 14 + rnd() * 12, 1.8);
  // antenna
  const ant = el('g', { id: 'antenna', style: 'transform-origin:246px 290px' }, head);
  el('ellipse', { cx: 246, cy: 290, rx: 9, ry: 11, fill: '#8f5f2a' }, ant);
  el('ellipse', { cx: 232, cy: 300, rx: 11, ry: 13, fill: '#a56d30' }, ant);
  const fun = el('g', { style: 'transform-origin:232px 310px' }, ant);
  el('ellipse', { cx: 224, cy: 328, rx: 12, ry: 20, fill: '#b47a38' }, fun);
  const arista = el('path', { d: 'M222,320 C205,300 190,285 165,262', stroke: '#2a1808', 'stroke-width': 2, fill: 'none' }, fun);
  for (let i = 1; i <= 7; i++) { const t = i / 8, x = 222 - 57 * t, y = 320 - 58 * t; el('path', { d: `M${x},${y} l${-6 - i},${-9} M${x},${y} l${6},${-10 - i}`, stroke: '#2a1808', 'stroke-width': 1.2, fill: 'none' }, fun); }
  // proboscis: rostrum + haustellum (rotates) + labellum (spreads)
  const prob = el('g', { id: 'proboscis', style: 'transform-origin:340px 355px' }, head);
  el('path', { d: 'M318,352 C320,372 330,380 342,382 C356,380 364,370 364,352 Z', fill: '#9a6a30' }, prob);                       // rostrum
  const haust = el('g', { id: 'haustellum', style: 'transform-origin:341px 378px', transform: 'rotate(-55 341 378)' }, prob);
  el('path', { d: 'M330,376 L326,430 C326,442 356,442 356,430 L352,376 Z', fill: 'url(#labG)', stroke: '#6a4218', 'stroke-width': 1.2 }, haust);
  const lab = el('g', { id: 'labellum', style: 'transform-origin:341px 440px' }, haust);
  el('path', { d: 'M326,436 C310,448 312,470 334,472 L348,472 C370,470 372,448 356,436 Z', fill: '#e6c68e', stroke: '#8a5a22', 'stroke-width': 1.2 }, lab);
  for (let i = 0; i < 6; i++) el('path', { d: `M${318 + i * 8},${444 + (i % 2) * 3} q3,10 ${(i - 2.5) * 2},${22}`, stroke: '#8a5a22', 'stroke-width': 0.9, fill: 'none', opacity: 0.8 }, lab);
  // ---- helpers
  const setT = (node, t) => node.setAttribute('transform', t);
  const anim = (ms, fn) => new Promise(res => { const t0 = performance.now(); (function step(t) { const k = Math.min(1, (t - t0) / ms); fn(k); if (k < 1) requestAnimationFrame(step); else res(); })(t0); });
  const ease = k => k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
  let busy = false, tBreath = 0;
  // ambient loop: breathing abdomen, antenna twitch, eye glint drift
  (function ambient(t) { tBreath = t / 1000; const b = 1 + 0.012 * Math.sin(tBreath * 2.4); setT(abd, `scale(${b} ${1 + 0.02 * Math.sin(tBreath * 2.4)})`); if (!busy) { setT(fun, `rotate(${3 * Math.sin(tBreath * 1.3) + 2 * Math.sin(tBreath * 7.1)} 232 310)`); glint.setAttribute('cx', 268 + 10 * Math.sin(tBreath * 0.5)); glint.setAttribute('cy', 205 + 6 * Math.cos(tBreath * 0.37)); } requestAnimationFrame(ambient); })(0);
  setInterval(() => { if (busy) return; const r = Math.random(); if (r < 0.35) { anim(600, k => setT(ant, `rotate(${-14 * Math.sin(k * Math.PI)} 246 290)`)); } else if (r < 0.5) { anim(500, k => setT(head, `rotate(${3 * Math.sin(k * Math.PI)} 420 300)`)); } else if (r < 0.62) { anim(420, k => setT(wing, `rotate(${-8 - 10 * Math.sin(k * Math.PI)} 600 245)`)); } }, 3000);
  async function act(bv) {
    if (!bv) return; busy = true; const tasks = [];
    if (bv.proboscis > 0.05) tasks.push(anim(1600, k => { const e = k < 0.3 ? ease(k / 0.3) : k > 0.75 ? ease((1 - k) / 0.25) : 1; setT(haust, `rotate(${-55 + 55 * e * Math.min(1, 0.4 + bv.proboscis)} 341 378)`); setT(lab, `scale(${1 + 0.35 * e} ${1 + 0.15 * e})`); }));
    if (bv.grooming > 0.05) tasks.push(anim(2200, k => { const s = Math.sin(k * Math.PI * 6) * 0.5 + 0.5, lift = ease(Math.min(1, k * 4)) * (k > 0.85 ? (1 - k) / 0.15 : 1); setT(legFront.fem, `rotate(${-95 * lift} 470 385)`); setT(legFront.tib, `rotate(${(-40 - 35 * s) * lift} 405 470)`); setT(legFront.tar, `rotate(${(-20 + 30 * s) * lift} 330 545)`); setT(ant, `rotate(${-10 * s * lift} 246 290)`); }));
    if (bv.escape > 0.05) tasks.push(anim(900, k => { const j = Math.sin(Math.min(1, k * 1.4) * Math.PI); wb.setAttribute('stdDeviation', 2.5 * (k < 0.9 ? 1 : 0)); setT(wing, `rotate(${-8 - 22 * Math.sin(k * 90) * (k < 0.9 ? 1 : 0)} 600 245)`); setT(body, `translate(0,${-90 * j * bv.escape}) rotate(${-10 * j} 470 400)`); [legHind, legMid].forEach(L => setT(L.fem, `rotate(${25 * j} ${L.fem.style.transformOrigin.replace(/px/g, '')})`)); if (k >= 1) { setT(body, ''); setT(wing, 'rotate(-8 600 245)'); [legHind, legMid].forEach(L => setT(L.fem, '')); } }));
    if (bv.walk_forward > 0.05 || bv.walk_backward > 0.05) tasks.push(anim(1200, k => { const ph = k * Math.PI * 4, dir = bv.walk_forward >= bv.walk_backward ? 1 : -1; [[legFront, 0], [legMid, Math.PI], [legHind, 0]].forEach(([L, o]) => { setT(L.fem, `rotate(${12 * dir * Math.sin(ph + o)} ${L.fem.style.transformOrigin.replace(/px/g, '')})`); setT(L.tib, `rotate(${8 * Math.sin(ph + o + 1)} ${L.tib.style.transformOrigin.replace(/px/g, '')})`); }); setT(thorax, `translate(0,${2 * Math.sin(ph * 2)})`); if (k >= 1) { [legFront, legMid, legHind].forEach(L => { setT(L.fem, ''); setT(L.tib, ''); }); setT(thorax, ''); } }));
    const turn = (bv.turn_right || 0) - (bv.turn_left || 0);
    if (Math.abs(turn) > 0.05) tasks.push(anim(900, k => { const s = Math.sin(k * Math.PI); setT(head, `rotate(${-18 * turn * s} 420 300) skewX(${8 * turn * s})`); }));
    if (bv.head > 0.05) tasks.push(anim(800, k => setT(head, `rotate(${6 * Math.sin(k * Math.PI * 3)} 420 300)`)));
    await Promise.all(tasks); busy = false;
  }
  // ---- odour particles and reinforcement flashes
  const fx = el('g', { id: 'fx' }, svg);
  async function smell(kind) {
    const color = kind === 'reward' ? '#ffd76a' : kind === 'punish' ? '#ff5a4a' : '#bfe3ff';
    const parts = [];
    for (let i = 0; i < 26; i++) { const c = el('circle', { cx: 40 + rnd() * 60, cy: 120 + rnd() * 320, r: 2 + rnd() * 3, fill: color, opacity: 0 }, fx); parts.push({ c, x: +c.getAttribute('cx'), y: +c.getAttribute('cy'), d: rnd() * 0.5 }); }
    const flash = kind !== 'none' ? el('ellipse', { cx: 330, cy: 265, rx: 140, ry: 140, fill: color, opacity: 0, filter: 'url(#soft)' }, fx) : null;
    await anim(2600, k => {
      parts.forEach(p => { const t = Math.max(0, Math.min(1, (k - p.d) / 0.6)); p.c.setAttribute('cx', p.x + (232 - p.x) * t + 6 * Math.sin(t * 12 + p.x)); p.c.setAttribute('cy', p.y + (318 - p.y) * t); p.c.setAttribute('opacity', t > 0 && t < 1 ? 0.75 * Math.sin(t * Math.PI) : 0); });
      setT(ant, `rotate(${-10 * Math.sin(k * Math.PI * 3)} 246 290)`);
      if (flash) flash.setAttribute('opacity', 0.35 * Math.sin(Math.min(1, k * 1.5) * Math.PI));
    });
    parts.forEach(p => p.c.remove()); if (flash) flash.remove(); setT(ant, '');
  }
  window.flyPortrait = { act, smell };
})();
