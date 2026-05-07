// Feedback client: vanilla JS, no framework.
// State lives in localStorage (rater_id + name). API is POST /api/feedback.

const LS_KEY_ID = 'pplgym.rater_id';
const LS_KEY_NAME = 'pplgym.rater_name';

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

// Modal-driven name capture. Resolves to the entered name (truthy) or '' (cancel).
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

  // Vote buttons
  const voteBtn = e.target.closest('.fb-btn[data-vote]');
  if (voteBtn) {
    const node = voteBtn.closest('.fb');
    if (!node) return;
    const name = await ensureName();
    if (!name) return; // cancelled
    updateNameUI(node);
    const { atomId, collection, datasetVersion } = widgetState(node);
    const rater_id = ensureRaterId();
    const vote = voteBtn.dataset.vote;
    setStatus(node, 'sending…');
    try {
      await postFeedback({
        atom_id: atomId, collection, dataset_version: datasetVersion,
        rater_id, rater_name: name, vote, comment: '',
      });
      setVoteHighlight(node, vote);
      setStatus(node, 'recorded ' + (vote === 'up' ? '👍' : '👎'), 'ok');
    } catch (err) {
      setStatus(node, 'error: ' + err.message, 'err');
    }
    return;
  }

  // Save-comment button
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
    setStatus(node, 'sending…');
    try {
      await postFeedback({
        atom_id: atomId, collection, dataset_version: datasetVersion,
        rater_id, rater_name: name, vote: 'neutral', comment,
      });
      ta.value = '';
      setStatus(node, 'comment saved', 'ok');
    } catch (err) {
      setStatus(node, 'error: ' + err.message, 'err');
    }
    return;
  }
});

// Initial render of name UI on every widget.
document.querySelectorAll('.fb').forEach(updateNameUI);
