// Feedback client: vanilla JS, no framework.
// State lives in localStorage (rater_id, name, and per-atom vote).
// API is POST /api/feedback. Submitting a comment carries forward the
// in-session vote so a single (rater, atom) ends up with one canonical row.

const LS_KEY_ID = 'pplgym.rater_id';
const LS_KEY_NAME = 'pplgym.rater_name';
const LS_VOTE_PREFIX = 'pplgym.vote.';

function ensureRaterId() {
  let id = localStorage.getItem(LS_KEY_ID);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(LS_KEY_ID, id);
  }
  return id;
}
function getName() { return localStorage.getItem(LS_KEY_NAME) || ''; }
function setName(n) { localStorage.setItem(LS_KEY_NAME, n); }

function getStoredVote(atomId) {
  return localStorage.getItem(LS_VOTE_PREFIX + atomId) || '';
}
function setStoredVote(atomId, vote) {
  if (vote === 'up' || vote === 'down') {
    localStorage.setItem(LS_VOTE_PREFIX + atomId, vote);
  } else {
    localStorage.removeItem(LS_VOTE_PREFIX + atomId);
  }
}

function promptForName(initial = '') {
  const back = document.getElementById('name-modal');
  if (!back) return Promise.resolve('');
  const input = document.getElementById('name-modal-input');
  const save = document.getElementById('name-modal-save');
  const cancel = document.getElementById('name-modal-cancel');
  input.value = initial;
  back.classList.add('open');
  setTimeout(() => input.focus(), 50);

  return new Promise((resolve) => {
    function close(value) {
      back.classList.remove('open');
      save.removeEventListener('click', onSave);
      cancel.removeEventListener('click', onCancel);
      input.removeEventListener('keydown', onKey);
      resolve(value);
    }
    function onSave() {
      const v = input.value.trim();
      if (!v) { input.focus(); return; }
      setName(v);
      close(v);
    }
    function onCancel() { close(''); }
    function onKey(e) {
      if (e.key === 'Enter') { e.preventDefault(); onSave(); }
      if (e.key === 'Escape') { e.preventDefault(); onCancel(); }
    }
    save.addEventListener('click', onSave);
    cancel.addEventListener('click', onCancel);
    input.addEventListener('keydown', onKey);
  });
}

async function ensureName() {
  const cur = getName();
  if (cur) return cur;
  return await promptForName();
}

async function postFeedback(body) {
  const r = await fetch('/api/feedback', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`server returned ${r.status}`);
  return await r.json();
}

function widgetState(node) {
  return {
    atomId: node.dataset.fbAtom,
    collection: node.dataset.fbCollection,
    datasetVersion: node.dataset.fbDatasetVersion,
  };
}

function updateNameUI(node) {
  const name = getName();
  const empty = node.querySelector('.fb-name-empty');
  const filled = node.querySelector('.fb-name-filled');
  if (!empty || !filled) return;
  if (name) {
    empty.hidden = true;
    filled.hidden = false;
    filled.querySelector('.fb-name-text').textContent = name;
  } else {
    empty.hidden = false;
    filled.hidden = true;
  }
}

function setStatus(node, msg, kind) {
  const el = node.querySelector('.fb-status');
  if (!el) return;
  el.textContent = msg;
  el.className = 'fb-status' + (kind ? ' is-' + kind : '');
}

function setVoteHighlight(node, vote) {
  const up = node.querySelector('.fb-btn.is-up');
  const down = node.querySelector('.fb-btn.is-down');
  if (!up || !down) return;
  up.classList.toggle('active', vote === 'up');
  down.classList.toggle('active', vote === 'down');
}

document.addEventListener('click', async (e) => {
  // Edit-name pencil
  const edit = e.target.closest('.fb-edit');
  if (edit) {
    e.preventDefault();
    const newName = await promptForName(getName());
    if (newName) {
      document.querySelectorAll('.fb').forEach(updateNameUI);
    }
    return;
  }

  // Vote buttons — toggle off if same button is clicked again.
  const voteBtn = e.target.closest('.fb-btn[data-vote]');
  if (voteBtn) {
    const node = voteBtn.closest('.fb');
    if (!node) return;
    const name = await ensureName();
    if (!name) return; // cancelled
    updateNameUI(node);
    const { atomId, collection, datasetVersion } = widgetState(node);
    const rater_id = ensureRaterId();
    const clicked = voteBtn.dataset.vote;
    const prior = getStoredVote(atomId);
    const vote = (prior === clicked) ? 'neutral' : clicked;
    const ta = node.querySelector('textarea');
    const comment = (ta?.value ?? '').trim();
    setStatus(node, 'sending…');
    try {
      await postFeedback({
        atom_id: atomId, collection, dataset_version: datasetVersion,
        rater_id, rater_name: name, vote, comment,
      });
      setStoredVote(atomId, vote);
      setVoteHighlight(node, vote);
      const label = vote === 'up' ? '👍' : vote === 'down' ? '👎' : '∅';
      setStatus(node, 'recorded ' + label + (comment ? ' + comment' : ''), 'ok');
    } catch (err) {
      setStatus(node, 'error: ' + err.message, 'err');
    }
    return;
  }

  // Save-comment button: carry the current vote forward.
  const submit = e.target.closest('.fb-submit');
  if (submit) {
    const node = submit.closest('.fb');
    if (!node) return;
    const ta = node.querySelector('textarea');
    const comment = (ta?.value ?? '').trim();
    if (!comment) {
      setStatus(node, 'comment is empty', 'err');
      return;
    }
    const name = await ensureName();
    if (!name) return;
    updateNameUI(node);
    const { atomId, collection, datasetVersion } = widgetState(node);
    const rater_id = ensureRaterId();
    const vote = getStoredVote(atomId) || 'neutral';
    setStatus(node, 'sending…');
    try {
      await postFeedback({
        atom_id: atomId, collection, dataset_version: datasetVersion,
        rater_id, rater_name: name, vote, comment,
      });
      ta.value = '';
      setStatus(node, 'comment saved' + (vote !== 'neutral' ? ' (carried ' + (vote === 'up' ? '👍' : '👎') + ')' : ''), 'ok');
    } catch (err) {
      setStatus(node, 'error: ' + err.message, 'err');
    }
    return;
  }
});

// Initial render: name UI + per-atom vote highlight from localStorage.
document.querySelectorAll('.fb').forEach((node) => {
  updateNameUI(node);
  const atomId = node.dataset.fbAtom;
  const v = getStoredVote(atomId);
  if (v) setVoteHighlight(node, v);
});
