/**
 * inf-mem content script engine, shared by every chat site.
 *
 * A site file (chatgpt.js, claude.js) passes startInfMem() a config of
 * selectors and a streaming check; everything else lives here:
 * - intercept the user's send (Enter or the send button), retrieve memories,
 *   show the review panel, and send the chosen text exactly once;
 * - after a send, wait for the reply that follows it to finish streaming,
 *   then extract memories from that prompt/reply pair exactly once.
 *
 * Listeners sit on the document in the capture phase and match targets by
 * selector, so they keep working when the site replaces the composer
 * (conversation switches re-render it). The engine's own synthetic send
 * events are untrusted and are ignored, so a send is never re-intercepted.
 */

const MEMORY_HEADER = '[Memory Context — inf-mem]';

// How long a finished reply's text must stay unchanged before extraction.
const SETTLE_MS = 600;

/**
 * Send a message to the background service worker
 * @param {string} type - Message type
 * @param {Object} data - Message payload
 * @returns {Promise<Object>} Response object from the background worker
 */
function sendToBackground(type, data = {}) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage({ type, ...data }, (response) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
      } else {
        resolve(response || {});
      }
    });
  });
}

/**
 * Prepend a memory context block to the prompt
 * @param {string} originalPrompt - The user's prompt
 * @param {Array<Object>} memories - Memories with {type, text}
 * @returns {string} The prompt, with the block if there are memories
 */
function injectMemoryContext(originalPrompt, memories) {
  if (!memories || memories.length === 0) {
    return originalPrompt;
  }
  const lines = memories.map((mem) => `• ${mem.type || 'UNKNOWN'}: ${mem.text || ''}`);
  return `${MEMORY_HEADER}\n${lines.join('\n')}\n[End Memory Context]\n\n${originalPrompt}`;
}

/**
 * Escape HTML special characters
 * @param {string} text - Text to escape
 * @returns {string} HTML-safe text
 */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/**
 * Show the memory review panel
 * @param {Array<Object>} memories - Memories that would be injected
 * @param {string|null} compressedPrompt - Compressed prompt preview, if any
 * @param {Object} actions - {withMemory, withoutMemory, compressed} callbacks
 * @returns {HTMLElement} The panel element
 */
function showPanel(memories, compressedPrompt, actions) {
  const panel = document.createElement('div');
  panel.className = 'inf-mem-panel';

  const items = memories
    .map((mem) => {
      const typeClass = mem.type ? mem.type.toLowerCase() : 'unknown';
      return `
    <div class="inf-mem-memory-item inf-mem-type-${typeClass}">
      <span class="inf-mem-type-badge">${escapeHtml(mem.type || 'UNKNOWN')}</span>
      <span class="inf-mem-text">${escapeHtml(mem.text)}</span>
    </div>`;
    })
    .join('');

  const compression = compressedPrompt
    ? `
    <div class="inf-mem-section">
      <div class="inf-mem-section-title">Compressed Prompt</div>
      <div class="inf-mem-compressed-preview">${escapeHtml(compressedPrompt.substring(0, 150))}...</div>
    </div>`
    : '';

  panel.innerHTML = `
    <div class="inf-mem-panel-content">
      <div class="inf-mem-header">
        <span class="inf-mem-title">📚 Memory Review</span>
        <button class="inf-mem-close-btn" data-action="close">&times;</button>
      </div>
      <div class="inf-mem-section">
        <div class="inf-mem-section-title">Injected Memories (${memories.length})</div>
        <div class="inf-mem-memory-list">${items}</div>
      </div>
      ${compression}
      <div class="inf-mem-buttons">
        <button class="inf-mem-btn inf-mem-btn-primary" data-action="with">Send with memory ✓</button>
        <button class="inf-mem-btn inf-mem-btn-secondary" data-action="without">Skip memory</button>
        ${compressedPrompt ? '<button class="inf-mem-btn inf-mem-btn-tertiary" data-action="compressed">Compress & Send</button>' : ''}
      </div>
    </div>`;

  document.body.appendChild(panel);
  setTimeout(() => panel.classList.add('visible'), 10);

  const handlers = {
    close: () => {},
    with: actions.withMemory,
    without: actions.withoutMemory,
    compressed: actions.compressed
  };
  // Buttons are found inside this panel, never by global id, so a panel
  // that is still fading out can't capture the next panel's clicks.
  panel.querySelectorAll('[data-action]').forEach((btn) => {
    btn.addEventListener('click', () => {
      hidePanel(panel);
      handlers[btn.dataset.action]();
    });
  });
  return panel;
}

/**
 * Hide and remove one panel
 * @param {HTMLElement} panel - The panel returned by showPanel
 * @returns {void}
 */
function hidePanel(panel) {
  panel.classList.remove('visible');
  setTimeout(() => panel.remove(), 300); // match the CSS transition
}

/**
 * Set text in a contenteditable composer so the site's editor sees it
 * @param {HTMLElement} el - The composer
 * @param {string} text - Text to set
 * @returns {void}
 */
function setComposerText(el, text) {
  el.focus();
  document.execCommand('selectAll', false, null);
  document.execCommand('insertText', false, text);
  el.dispatchEvent(new Event('input', { bubbles: true }));
}

/**
 * Run inf-mem on a chat site
 * @param {Object} site - Site config:
 *   name: string for logs
 *   composer: CSS selector for the prompt editor
 *   sendButton: CSS selector for the send button
 *   assistantMessages: CSS selector matching every assistant reply, in order
 *   isStreaming(): true while a reply is still being written
 *   replyText(el): the text of one reply element
 * @returns {void}
 */
function startInfMem(site) {
  let busy = false;
  // The prompt waiting for its reply: {prompt, repliesBefore}
  let pending = null;
  let settleTimer = null;

  const composer = () => document.querySelector(site.composer);
  const replies = () => document.querySelectorAll(site.assistantMessages);

  async function intercept(prompt) {
    busy = true;
    try {
      const result = await sendToBackground('RETRIEVE', { prompt });
      const memories = result.memories || [];
      if (result.error || memories.length === 0) {
        await send(prompt, prompt);
        return;
      }

      let compressed = null;
      const { settings } = await sendToBackground('GET_SETTINGS', {});
      if (settings?.compressionEnabled) {
        compressed = (await sendToBackground('COMPRESS', { prompt })).compressed || null;
      }

      showPanel(memories, compressed, {
        withMemory: () => send(injectMemoryContext(prompt, memories), prompt),
        withoutMemory: () => send(prompt, prompt),
        compressed: () => send(injectMemoryContext(compressed, memories), prompt)
      });
    } catch (error) {
      console.error(`[inf-mem] ${site.name}: intercept failed, sending as typed`, error);
      await send(prompt, prompt);
    } finally {
      busy = false;
    }
  }

  async function send(finalText, originalPrompt) {
    const el = composer();
    if (!el) {
      console.error(`[inf-mem] ${site.name}: composer not found`);
      return;
    }
    pending = { prompt: originalPrompt, repliesBefore: replies().length };
    setComposerText(el, finalText);
    await new Promise((resolve) => setTimeout(resolve, 50));

    // Untrusted events: our own listeners ignore them (see onKeydown/onClick).
    const button = document.querySelector(site.sendButton);
    if (button && !button.disabled) {
      button.click();
    } else {
      el.dispatchEvent(new KeyboardEvent('keydown', {
        key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true, cancelable: true
      }));
    }
  }

  function takePrompt(e) {
    if (!e.isTrusted || busy) return null;
    const el = composer();
    const prompt = el?.innerText?.trim();
    if (!prompt) return null;
    e.preventDefault();
    e.stopImmediatePropagation();
    return prompt;
  }

  function onKeydown(e) {
    if (e.key !== 'Enter' || e.shiftKey || e.isComposing) return;
    if (!e.target.closest?.(site.composer)) return;
    const prompt = takePrompt(e);
    if (prompt) intercept(prompt);
  }

  function onClick(e) {
    if (!e.target.closest?.(site.sendButton)) return;
    const prompt = takePrompt(e);
    if (prompt) intercept(prompt);
  }

  // Extract once the reply that followed the pending prompt is complete and
  // its text has stopped changing for SETTLE_MS.
  function checkReply() {
    if (!pending) return;
    const all = replies();
    if (all.length <= pending.repliesBefore || site.isStreaming()) return;
    const text = site.replyText(all[pending.repliesBefore]).trim();
    if (!text) return;

    clearTimeout(settleTimer);
    settleTimer = setTimeout(() => {
      const now = replies();
      if (!pending || site.isStreaming() || now.length <= pending.repliesBefore) return;
      if (site.replyText(now[pending.repliesBefore]).trim() !== text) return;
      const { prompt } = pending;
      pending = null;
      sendToBackground('EXTRACT', { prompt, response: text }).catch((error) => {
        console.error(`[inf-mem] ${site.name}: extract failed`, error);
      });
    }, SETTLE_MS);
  }

  document.addEventListener('keydown', onKeydown, true);
  document.addEventListener('click', onClick, true);
  new MutationObserver(checkReply).observe(document.body, {
    childList: true, subtree: true, characterData: true, attributes: true
  });
}
