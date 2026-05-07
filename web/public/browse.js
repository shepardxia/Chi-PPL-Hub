const atoms = Array.from(document.querySelectorAll('.atom'));
const groups = Array.from(document.querySelectorAll('.group'));
const main = document.querySelector('main');
const search = document.getElementById('search');
const sortSel = document.getElementById('sort');
const bkPills = Array.from(document.querySelectorAll('.pill[data-bk]'));
const visibleCount = document.getElementById('visible-count');
const emptyMsg = document.getElementById('empty-msg');

const BUCKET_ORDER = {
  'fail': 0, 'shape!': 1, 'TV=1': 2, 'val-': 3,
  'TV<1': 4, 'TV<.5': 5, 'TV<.05': 6, 'TV=0': 7, 'val+': 8, 'no-run': 9,
};

function activeSet(pills, attr) {
  const set = new Set();
  for (const p of pills) if (p.classList.contains('active')) set.add(p.dataset[attr]);
  return set;
}

function applyFilter() {
  const q = (search?.value ?? '').trim().toLowerCase();
  const bk = activeSet(bkPills, 'bk');
  const isFiltering = q !== '' || bk.size > 0;
  let visible = 0;
  for (const a of atoms) {
    const okBk = bk.size === 0 || bk.has(a.dataset.bk);
    const okQ = !q || (a.dataset.search ?? '').indexOf(q) !== -1;
    const show = okBk && okQ;
    a.classList.toggle('is-hidden', !show);
    if (show) visible++;
  }
  for (const g of groups) {
    const matched = g.querySelectorAll('.atom:not(.is-hidden)').length;
    g.classList.toggle('is-hidden', matched === 0 && isFiltering);
    if (isFiltering) g.open = matched > 0;
  }
  if (visibleCount) visibleCount.textContent = visible;
  if (emptyMsg) emptyMsg.style.display = visible === 0 ? '' : 'none';
}

function applySort() {
  if (!sortSel || !main) return;
  const mode = sortSel.value;
  const sorted = [...atoms];
  if (mode === 'bucket') {
    sorted.sort((a, b) => (BUCKET_ORDER[a.dataset.bk] ?? 99) - (BUCKET_ORDER[b.dataset.bk] ?? 99)
                          || a.dataset.aid.localeCompare(b.dataset.aid));
  } else if (mode === 'tv-desc' || mode === 'tv-asc') {
    const dir = mode === 'tv-desc' ? -1 : 1;
    sorted.sort((a, b) => {
      const av = a.dataset.tv === '' ? null : parseFloat(a.dataset.tv);
      const bv = b.dataset.tv === '' ? null : parseFloat(b.dataset.tv);
      if (av === null && bv === null) return a.dataset.aid.localeCompare(b.dataset.aid);
      if (av === null) return 1;
      if (bv === null) return -1;
      return dir * (av - bv) || a.dataset.aid.localeCompare(b.dataset.aid);
    });
  } else {
    sorted.sort((a, b) => a.dataset.aid.localeCompare(b.dataset.aid));
  }
  for (const a of sorted) main.appendChild(a);
}

for (const p of bkPills) p.addEventListener('click', () => { p.classList.toggle('active'); applyFilter(); });
search?.addEventListener('input', applyFilter);
search?.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') { e.target.blur(); search.value = ''; applyFilter(); }
});
sortSel?.addEventListener('change', applySort);

document.getElementById('expand-all')?.addEventListener('click', () => { for (const g of groups) g.open = true; });
document.getElementById('collapse-all')?.addEventListener('click', () => { for (const g of groups) g.open = false; });

document.addEventListener('click', (e) => {
  const pill = e.target.closest('.run-pill');
  if (!pill) return;
  const atom = pill.closest('.atom');
  if (!atom) return;
  const target = pill.dataset.run;
  for (const p of atom.querySelectorAll('.run-pill')) p.classList.toggle('active', p.dataset.run === target);
  for (const p of atom.querySelectorAll('.run-panel')) p.classList.toggle('active', p.dataset.run === target);
});

function focusFromHash() {
  const h = location.hash.slice(1);
  if (!h) return;
  const t = document.getElementById(h);
  if (!t) return;
  for (const a of atoms) a.classList.remove('is-target');
  t.classList.add('is-target');
  const grp = t.closest('.group');
  if (grp) grp.open = true;
  t.open = true;
  t.scrollIntoView({ block: 'start' });
}
window.addEventListener('hashchange', focusFromHash);
for (const a of atoms) {
  a.addEventListener('toggle', () => { if (a.open) history.replaceState(null, '', '#' + a.id); });
}

const visibleAtoms = () => atoms.filter((a) => !a.classList.contains('is-hidden'));
function currentIndex(list) { for (let i = 0; i < list.length; i++) if (list[i].classList.contains('is-target')) return i; return -1; }
function focusAtom(a) {
  for (const x of atoms) x.classList.remove('is-target');
  a.classList.add('is-target');
  a.scrollIntoView({ block: 'center', behavior: 'smooth' });
  history.replaceState(null, '', '#' + a.id);
}
document.addEventListener('keydown', (e) => {
  if (e.target === search || e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') {
    if (e.key === 'Escape') e.target.blur();
    return;
  }
  if (e.key === '/') { e.preventDefault(); search?.focus(); search?.select(); return; }
  if (e.key === 'Escape') {
    document.querySelectorAll('.atom.is-target').forEach((a) => a.classList.remove('is-target'));
    return;
  }
  const list = visibleAtoms();
  if (list.length === 0) return;
  if (e.key === 'j' || e.key === 'ArrowDown') {
    e.preventDefault();
    const i = currentIndex(list);
    focusAtom(list[Math.min(i + 1, list.length - 1)] || list[0]);
  } else if (e.key === 'k' || e.key === 'ArrowUp') {
    e.preventDefault();
    const i = currentIndex(list);
    focusAtom(list[Math.max(i - 1, 0)] || list[0]);
  } else if (e.key === 'Enter') {
    const i = currentIndex(list);
    if (i >= 0) list[i].open = !list[i].open;
  }
});

focusFromHash();
