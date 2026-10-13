// Shared behaviour for the chat-site fixtures. Not a capture of either live
// site: it reproduces only the elements and streaming signals the content
// scripts depend on. Checking them against the live sites is a separate,
// dated step (docs/live-checks/).
//
// window.__submits records every message the "site" accepted.
// A reply streams in CHUNKS pieces, CHUNK_MS apart.
(() => {
  const SITE = document.body.dataset.site;
  const CHUNKS = 40;
  const CHUNK_MS = 25;
  window.__submits = [];
  let replyCount = 0;

  const thread = document.getElementById('thread');
  const composer = () => document.querySelector('[data-fixture="composer"]');
  const sendBtn = document.getElementById('send');

  function addReply(n, finished) {
    const words = Array.from({ length: CHUNKS }, (_, i) => `r${n}w${i}`);
    let el, body;
    if (SITE === 'chatgpt') {
      el = document.createElement('div');
      el.setAttribute('data-message-author-role', 'assistant');
      body = el;
    } else {
      el = document.createElement('div');
      el.setAttribute('data-is-streaming', finished ? 'false' : 'true');
      body = document.createElement('div');
      body.className = 'font-claude-message';
      el.appendChild(body);
    }
    thread.appendChild(el);
    return { el, body, words };
  }

  function setStreaming(on, el) {
    if (SITE === 'chatgpt') {
      sendBtn.hidden = on;
      const stop = document.getElementById('stop');
      if (on && !stop) {
        const b = document.createElement('button');
        b.id = 'stop';
        b.setAttribute('data-testid', 'stop-button');
        sendBtn.after(b);
      } else if (!on && stop) {
        stop.remove();
      }
    } else {
      el.setAttribute('data-is-streaming', on ? 'true' : 'false');
    }
  }

  function submit() {
    const text = composer().innerText.trim();
    if (!text) return;
    window.__submits.push(text);
    composer().innerHTML = '';
    const user = document.createElement('div');
    user.className = 'user-message';
    user.textContent = text;
    thread.appendChild(user);

    const n = ++replyCount;
    const { el, body, words } = addReply(n, false);
    setStreaming(true, el);
    let i = 0;
    const timer = setInterval(() => {
      body.textContent += (i ? ' ' : '') + words[i];
      i += 1;
      if (i === words.length) {
        clearInterval(timer);
        setStreaming(false, el);
      }
    }, CHUNK_MS);
  }

  // Finished replies already in the thread (?history=N).
  const history = Number(new URLSearchParams(location.search).get('history') || 0);
  for (let n = 1; n <= history; n++) {
    const { body, words } = addReply(++replyCount, true);
    body.textContent = words.join(' ');
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && e.target.closest('[data-fixture="composer"]')) {
      e.preventDefault();
      submit();
    }
  });
  sendBtn.addEventListener('click', submit);

  // A conversation switch re-renders the composer as a new node.
  window.__swapComposer = () => {
    const old = composer();
    old.replaceWith(old.cloneNode(false));
  };
})();
