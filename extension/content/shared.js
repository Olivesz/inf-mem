/**
 * inf-mem Shared Content Script Utilities
 *
 * Provides common utilities for both ChatGPT and Claude content scripts:
 * - Message routing to background service worker
 * - Memory context injection into prompts
 * - Review panel UI management
 * - User ID extraction
 *
 * Used by both content/chatgpt.js and content/claude.js
 */

const BACKEND_PORT = 7842;

/**
 * Send a message to the background service worker
 * @param {string} type - Message type (RETRIEVE, EXTRACT, COMPRESS, GET_SETTINGS, SAVE_SETTINGS)
 * @param {Object} data - Message data payload
 * @description Wraps chrome.runtime.sendMessage with Promise-based interface
 * @returns {Promise<Object>} Response object from background script
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
 * Inject memory context into the original prompt
 * @param {string} originalPrompt - The user's original prompt text
 * @param {Array<Object>} memories - Array of memory objects with {type, text, ...}
 * @description Formats memories as a context block prepended to the prompt
 * @returns {string} Prompt with memory context injected
 *
 * Format:
 * [Memory Context — inf-mem]
 * • FACT: {text}
 * • PROJECT: {text}
 * ...
 * [End Memory Context]
 *
 * {originalPrompt}
 */
function injectMemoryContext(originalPrompt, memories) {
  if (!memories || memories.length === 0) {
    return originalPrompt;
  }

  const memoryLines = memories.map((mem) => {
    const type = mem.type || 'UNKNOWN';
    const text = mem.text || '';
    return `• ${type}: ${text}`;
  });

  const contextBlock = `[Memory Context — inf-mem]
${memoryLines.join('\n')}
[End Memory Context]`;

  return `${contextBlock}\n\n${originalPrompt}`;
}

/**
 * Show the memory review panel
 * @param {Array<Object>} memories - Retrieved memories to display
 * @param {string|null} compressedPrompt - Optional compressed prompt preview
 * @param {string} originalPrompt - The original user prompt
 * @param {Function} onSendWithContext - Callback when user clicks "Send with memory"
 * @param {Function} onSendWithout - Callback when user clicks "Skip memory"
 * @param {Function} onCompress - Callback when user clicks "Compress & Send"
 * @description Creates and injects review panel into DOM, slides in from right
 * @returns {void}
 */
function showPanel(memories, compressedPrompt, originalPrompt, onSendWithContext, onSendWithout, onCompress) {
  // Remove any existing panel
  hidePanel();

  // Create panel container
  const panel = document.createElement('div');
  panel.id = 'inf-mem-panel';
  panel.className = 'inf-mem-panel';

  // Build memory items HTML
  const memoryListHTML = memories
    .map((mem) => {
      const typeClass = mem.type ? mem.type.toLowerCase() : 'unknown';
      return `
    <div class="inf-mem-memory-item inf-mem-type-${typeClass}">
      <span class="inf-mem-type-badge">${mem.type || 'UNKNOWN'}</span>
      <span class="inf-mem-text">${escapeHtml(mem.text)}</span>
    </div>
  `;
    })
    .join('');

  // Build compression preview section (if available)
  const compressionPreview = compressedPrompt
    ? `
    <div class="inf-mem-section">
      <div class="inf-mem-section-title">Compressed Prompt</div>
      <div class="inf-mem-compressed-preview">${escapeHtml(compressedPrompt.substring(0, 150))}...</div>
    </div>
  `
    : '';

  // Panel HTML structure
  panel.innerHTML = `
    <div class="inf-mem-panel-content">
      <div class="inf-mem-header">
        <span class="inf-mem-title">📚 Memory Review</span>
        <button class="inf-mem-close-btn" id="inf-mem-close">&times;</button>
      </div>

      <div class="inf-mem-section">
        <div class="inf-mem-section-title">
          Injected Memories (${memories.length})
        </div>
        <div class="inf-mem-memory-list">
          ${memoryListHTML || '<div style="color: #999; font-size: 12px;">No memories found</div>'}
        </div>
      </div>

      ${compressionPreview}

      <div class="inf-mem-buttons">
        <button class="inf-mem-btn inf-mem-btn-primary" id="inf-mem-send-with">
          Send with memory ✓
        </button>
        <button class="inf-mem-btn inf-mem-btn-secondary" id="inf-mem-send-without">
          Skip memory
        </button>
        ${compressedPrompt ? `
          <button class="inf-mem-btn inf-mem-btn-tertiary" id="inf-mem-compress">
            Compress & Send
          </button>
        ` : ''}
      </div>
    </div>
  `;

  // Inject into page
  document.body.appendChild(panel);

  // Add visible class for animation after a tick (allows CSS transition)
  setTimeout(() => {
    panel.classList.add('visible');
  }, 10);

  // Attach event listeners
  document.getElementById('inf-mem-close')?.addEventListener('click', hidePanel);
  document.getElementById('inf-mem-send-with')?.addEventListener('click', () => {
    hidePanel();
    onSendWithContext();
  });
  document.getElementById('inf-mem-send-without')?.addEventListener('click', () => {
    hidePanel();
    onSendWithout();
  });
  document.getElementById('inf-mem-compress')?.addEventListener('click', () => {
    hidePanel();
    onCompress();
  });
}

/**
 * Hide and remove the memory review panel
 * @description Removes panel from DOM with cleanup
 * @returns {void}
 */
function hidePanel() {
  const panel = document.getElementById('inf-mem-panel');
  if (panel) {
    panel.classList.remove('visible');
    setTimeout(() => {
      panel?.remove?.();
    }, 300); // Match CSS animation duration
  }
}

/**
 * Extract the current user ID
 * @description Currently returns "default" as placeholder for future per-user support
 * @returns {string} User ID identifier
 */
function extractUserId() {
  // Placeholder for future per-user support (could check page for username, etc.)
  return 'default';
}

/**
 * Helper: Escape HTML special characters to prevent injection
 * @param {string} text - Text to escape
 * @description Prevents XSS by escaping HTML entities
 * @returns {string} Escaped HTML-safe text
 */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
