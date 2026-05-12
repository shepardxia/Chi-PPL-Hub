// ppl-gym browse: hash-based atom selection, filtering, keyboard nav,
// code-pane source switching.

const BUCKET_GLYPH = {
  'TV=0': '●', 'TV<.05': '○', 'TV<.5': '◐', 'TV<1': '◑',
  'TV=1': '■', 'val+': '✓', 'val-': '✗', 'shape!': '!',
  'fail': '×', 'no-run': '◌',
};

const items = Array.from(document.querySelectorAll('.item'));
const groupHeads = Array.from(document.querySelectorAll('.group-head'));
const atoms = Array.from(document.querySelectorAll('.atom'));
const search = document.getElementById('search');
const bucketPills = Array.from(document.querySelectorAll('.pill[data-bk]'));
const visibleCount = document.getElementById('visible-count');
const detailEmpty = document.getElementById('detail-empty');
const legendToggle = document.getElementById('legend-toggle');
const legendPanel = document.getElementById('legend-panel');
const detailEl = document.getElementById('detail');
const listScroll = document.getElementById('list-scroll');

// ------------------------------------------------------------------
// Filtering
// ------------------------------------------------------------------

function activeBuckets() {
  const out = new Set();
  for (const p of bucketPills) if (p.classList.contains('active')) out.add(p.dataset.bk);
  return out;
}

function applyFilter() {
  const q = (search?.value ?? '').trim().toLowerCase();
  const bk = activeBuckets();
  let visible = 0;
  for (const item of items) {
    const okBk = bk.size === 0 || bk.has(item.dataset.bk);
    const okQ = !q || (item.dataset.search ?? '').indexOf(q) !== -1;
    const show = okBk && okQ;
    item.classList.toggle('is-hidden', !show);
    if (show) visible++;
  }
  for (const gh of groupHeads) {
    const matched = document.querySelectorAll(
      `.item[data-group="${cssEscape(gh.dataset.group)}"]:not(.is-hidden)`
    ).length;
    gh.classList.toggle('is-hidden', matched === 0);
  }
  if (visibleCount) visibleCount.textContent = visible;
}

function cssEscape(s) {
  return String(s).replace(/(["\\])/g, '\\$1');
}

bucketPills.forEach((p) => p.addEventListener('click', () => {
  p.classList.toggle('active');
  applyFilter();
}));

search?.addEventListener('input', applyFilter);
search?.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') { e.target.blur(); search.value = ''; applyFilter(); }
});

// ------------------------------------------------------------------
// Selection (hash-based)
// ------------------------------------------------------------------

function showAtom(domId) {
  let any = false;
  for (const a of atoms) {
    const match = a.id === domId;
    a.classList.toggle('is-active', match);
    if (match) any = true;
  }
  for (const it of items) it.classList.toggle('is-selected', it.dataset.aid && domIdFor(it.dataset.aid) === domId);
  if (detailEmpty) detailEmpty.style.display = any ? 'none' : '';
  if (any && detailEl) detailEl.scrollTop = 0;
}

function domIdFor(atomId) {
  return 'atom-' + String(atomId).replace(/\//g, '--').replace(/ /g, '_');
}

function focusFromHash() {
  const h = location.hash.slice(1);
  if (!h) { showAtom(''); return; }
  showAtom(h);
  // Scroll the selected item into view in the rail.
  const sel = document.querySelector('.item.is-selected');
  sel?.scrollIntoView({ block: 'center' });
}

window.addEventListener('hashchange', focusFromHash);

items.forEach((it) => it.addEventListener('click', () => {
  const aid = it.dataset.aid;
  if (!aid) return;
  const newHash = '#' + domIdFor(aid);
  if (location.hash === newHash) {
    // Re-render in case scroll position drifted.
    focusFromHash();
  } else {
    history.pushState(null, '', newHash);
    focusFromHash();
  }
}));

// ------------------------------------------------------------------
// Code-pane source switching
// ------------------------------------------------------------------

function paneStatusEl(pane) {
  return pane.querySelector('.codepane-status');
}

function applyPaneSelection(pane, key) {
  for (const pre of pane.querySelectorAll('[data-src]')) {
    pre.hidden = pre.dataset.src !== key;
  }
  // Update status badge from atom's data-sources map.
  const atom = pane.closest('.atom');
  if (!atom) return;
  let map = {};
  try { map = JSON.parse(atom.dataset.sources || '{}'); } catch {}
  const info = map[key] || { bucket: 'no-run', tone: 'muted' };
  const statusEl = paneStatusEl(pane);
  if (statusEl) {
    statusEl.className = 'codepane-status tone-' + (info.tone || 'muted');
    const tvStr = info.tv != null ? ' · TV=' + Number(info.tv).toFixed(2) : '';
    statusEl.innerHTML =
      '<span>' + (BUCKET_GLYPH[info.bucket] || '◌') + '</span>' +
      '<span>' + info.bucket + '</span>' +
      (tvStr ? '<span>' + tvStr + '</span>' : '');
  }
}

document.addEventListener('change', (e) => {
  const sel = e.target.closest && e.target.closest('.codepane-select');
  if (!sel) return;
  const pane = sel.closest('.codepane');
  if (!pane) return;
  applyPaneSelection(pane, sel.value);
});

// Run-tab chip click → set right pane to that run.
document.addEventListener('click', (e) => {
  const chip = e.target.closest && e.target.closest('.run-tab[data-run]');
  if (!chip) return;
  const atom = chip.closest('.atom');
  if (!atom) return;
  const sel = atom.querySelector('.codepane[data-pane="right"] .codepane-select');
  if (!sel) return;
  sel.value = chip.dataset.run;
  sel.dispatchEvent(new Event('change', { bubbles: true }));
});

// ------------------------------------------------------------------
// Legend popover
// ------------------------------------------------------------------

legendToggle?.addEventListener('click', (e) => {
  e.stopPropagation();
  if (!legendPanel) return;
  legendPanel.hidden = !legendPanel.hidden;
});
document.addEventListener('click', (e) => {
  if (!legendPanel || legendPanel.hidden) return;
  if (e.target === legendToggle) return;
  if (!legendPanel.contains(e.target)) legendPanel.hidden = true;
});

// ------------------------------------------------------------------
// Keyboard navigation
// ------------------------------------------------------------------

function visibleItems() {
  return items.filter((i) => !i.classList.contains('is-hidden'));
}

function currentSelectedIndex(list) {
  for (let i = 0; i < list.length; i++) if (list[i].classList.contains('is-selected')) return i;
  return -1;
}

function jumpItem(delta) {
  const list = visibleItems();
  if (list.length === 0) return;
  const cur = currentSelectedIndex(list);
  const next = cur < 0 ? 0 : Math.max(0, Math.min(list.length - 1, cur + delta));
  const target = list[next];
  if (!target) return;
  const aid = target.dataset.aid;
  if (!aid) return;
  location.hash = '#' + domIdFor(aid);
}

document.addEventListener('keydown', (e) => {
  const tag = e.target.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
    if (e.key === 'Escape') e.target.blur();
    return;
  }
  if (e.key === '/') { e.preventDefault(); search?.focus(); search?.select(); return; }
  if (e.key === 'Escape') return;
  if (e.key === 'j' || e.key === 'ArrowDown') { e.preventDefault(); jumpItem(+1); }
  else if (e.key === 'k' || e.key === 'ArrowUp') { e.preventDefault(); jumpItem(-1); }
});

// ------------------------------------------------------------------
// Initial render
// ------------------------------------------------------------------

applyFilter();
focusFromHash();
