/**
 * inf-mem Popup Script
 *
 * Handles the extension popup functionality:
 * - Displays memory count and last updated time
 * - Manages compression toggle setting
 * - Shows all stored memories with type badges and importance dots
 * - Allows deletion of individual memories
 * - Supports clearing all memories with confirmation
 * - Polls backend health status (updates every 10 seconds)
 */

let healthCheckInterval = null;

/**
 * Initialize popup on load
 * @description Runs all initialization tasks in sequence
 * @returns {Promise<void>}
 */
async function initPopup() {
  console.log('[inf-mem-popup] Initializing popup');

  try {
    // Check backend health
    await updateHealthStatus();

    // Load and display memories
    await loadAndRenderMemories();

    // Load compression setting
    await loadCompressionSetting();

    // Attach event listeners
    attachEventListeners();

    // Start health check polling
    startHealthCheckPolling();
  } catch (error) {
    console.error('[inf-mem-popup] Init error:', error);
    updateFooterMessage('Error loading popup');
  }
}

/**
 * Check backend health and update status indicator
 * @description Makes health check request, updates status dot and text
 * @returns {Promise<void>}
 */
async function updateHealthStatus() {
  try {
    const response = await sendToBackgroundPopup('HEALTH_CHECK', {});
    const isOnline = response.online;

    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');

    if (isOnline) {
      statusDot.classList.add('online');
      statusDot.classList.remove('offline');
      statusText.textContent = 'Connected';
    } else {
      statusDot.classList.add('offline');
      statusDot.classList.remove('online');
      statusText.textContent = 'Offline';
      updateFooterMessage('Backend offline. Start with: make start');
    }
  } catch (error) {
    console.error('[inf-mem-popup] Health check error:', error);
    const statusDot = document.getElementById('statusDot');
    statusDot.classList.add('offline');
  }
}

/**
 * Load and render all stored memories
 * @description Fetches the user's memories from the backend via the background worker
 * @returns {Promise<void>}
 */
async function loadAndRenderMemories() {
  const response = await sendToBackgroundPopup('GET_MEMORIES', {});
  const memories = response.memories || [];
  if (response.error) {
    updateFooterMessage(response.error);
  }

  document.getElementById('memoryCount').textContent = memories.length;

  const latest = memories
    .map((m) => m.updated_at || m.created_at)
    .filter(Boolean)
    .sort()
    .pop();
  document.getElementById('lastUpdated').textContent = latest ? formatDate(latest) : '—';

  renderMemories(memories);
}

/**
 * Render memory cards in the popup
 * @param {Array<Object>} memories - Array of memory objects to display
 * @description Groups memories by type and renders them as cards
 * @returns {void}
 */
function renderMemories(memories) {
  const memoriesList = document.getElementById('memoriesList');

  if (!memories || memories.length === 0) {
    memoriesList.innerHTML = `
      <div class="inf-mem-empty-state">
        <div class="inf-mem-empty-icon">📭</div>
        <div class="inf-mem-empty-text">No memories yet</div>
        <div class="inf-mem-empty-desc">Memories will appear as you chat</div>
      </div>
    `;
    return;
  }

  // Group memories by type
  const grouped = {};
  memories.forEach((mem) => {
    const type = mem.type || 'unknown';
    if (!grouped[type]) {
      grouped[type] = [];
    }
    grouped[type].push(mem);
  });

  // Render cards
  let html = '';
  Object.entries(grouped).forEach(([type, mems]) => {
    mems.forEach((mem) => {
      const typeClass = type.toLowerCase();
      const importanceDots = renderImportanceDots(mem.importance || 5);

      html += `
        <div class="inf-mem-memory-card type-${typeClass}">
          <div class="inf-mem-memory-header">
            <span class="inf-mem-memory-type ${typeClass}">${type}</span>
            <button class="inf-mem-delete-btn" data-id="${mem.id || ''}" title="Delete memory">
              ✕
            </button>
          </div>
          <div class="inf-mem-memory-text">${escapeHtmlPopup(mem.text)}</div>
          <div class="inf-mem-importance-dots">${importanceDots}</div>
        </div>
      `;
    });
  });

  memoriesList.innerHTML = html;

  // Attach delete button listeners
  memoriesList.querySelectorAll('.inf-mem-delete-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const memoryId = btn.getAttribute('data-id');
      if (memoryId) {
        deleteMemory(memoryId);
      }
    });
  });
}

/**
 * Render importance dots (1-10 scale as 5 dots)
 * @param {number} importance - Importance value from 1-10
 * @description Each dot represents 2 importance points
 * @returns {string} HTML string for importance dots
 */
function renderImportanceDots(importance) {
  const filledDots = Math.ceil(importance / 2);
  let html = '';

  for (let i = 0; i < 5; i++) {
    html += `<div class="inf-mem-dot ${i < filledDots ? 'filled' : ''}"></div>`;
  }

  return html;
}

/**
 * Load compression setting from storage
 * @description Gets setting and updates toggle switch state
 * @returns {Promise<void>}
 */
async function loadCompressionSetting() {
  try {
    const settings = await sendToBackgroundPopup('GET_SETTINGS', {});
    const compressionEnabled = settings.settings?.compressionEnabled || false;

    const toggle = document.getElementById('compressionToggle');
    if (toggle) {
      toggle.checked = compressionEnabled;
    }
  } catch (error) {
    console.error('[inf-mem-popup] Load compression setting error:', error);
  }
}

/**
 * Attach event listeners to UI elements
 * @description Sets up click handlers for toggle and buttons
 * @returns {void}
 */
function attachEventListeners() {
  // Compression toggle
  document.getElementById('compressionToggle')?.addEventListener('change', (e) => {
    saveCompressionSetting(e.target.checked);
  });

  // Clear all button
  document.getElementById('clearAllBtn')?.addEventListener('click', () => {
    if (confirm('Clear all memories? This cannot be undone.')) {
      clearAllMemories();
    }
  });
}

/**
 * Save compression setting to storage
 * @param {boolean} enabled - Whether compression is enabled
 * @description Sends setting to background and updates storage
 * @returns {Promise<void>}
 */
async function saveCompressionSetting(enabled) {
  try {
    await sendToBackgroundPopup('SAVE_SETTINGS', {
      settings: { compressionEnabled: enabled }
    });
    console.log('[inf-mem-popup] Compression setting saved:', enabled);
  } catch (error) {
    console.error('[inf-mem-popup] Save compression error:', error);
  }
}

/**
 * Delete a single memory
 * @param {string} memoryId - ID of memory to delete
 * @description Deletes it on the backend, then re-renders from the backend
 * @returns {Promise<void>}
 */
async function deleteMemory(memoryId) {
  const response = await sendToBackgroundPopup('DELETE_MEMORY', { id: memoryId });
  if (!response.success) {
    updateFooterMessage(`Delete failed: ${response.error || 'unknown error'}`);
  }
  await loadAndRenderMemories();
}

/**
 * Clear all memories (the user has already confirmed)
 * @description Deletes every memory for this user on the backend, then re-renders
 * @returns {Promise<void>}
 */
async function clearAllMemories() {
  const response = await sendToBackgroundPopup('CLEAR_MEMORIES', {});
  updateFooterMessage(
    response.success ? `Cleared ${response.deleted} memories` : `Clear failed: ${response.error || 'unknown error'}`
  );
  await loadAndRenderMemories();
}

/**
 * Start polling backend health status
 * @description Checks health every 10 seconds and updates status indicator
 * @returns {void}
 */
function startHealthCheckPolling() {
  // Clear any existing interval
  if (healthCheckInterval) {
    clearInterval(healthCheckInterval);
  }

  // Poll every 10 seconds
  healthCheckInterval = setInterval(() => {
    updateHealthStatus();
  }, 10000);
}

/**
 * Update footer message
 * @param {string} message - Message to display
 * @description Shows a temporary or persistent footer message
 * @returns {void}
 */
function updateFooterMessage(message) {
  const footerMessage = document.getElementById('footerMessage');
  if (footerMessage) {
    footerMessage.textContent = message;
  }
}

/**
 * Format date for display
 * @param {string|Date} date - Date to format
 * @description Converts to readable format (e.g., "2 hours ago")
 * @returns {string} Formatted date string
 */
function formatDate(date) {
  if (!date) return '—';

  const dateObj = typeof date === 'string' ? new Date(date) : date;
  const now = new Date();
  const diffMs = now - dateObj;
  const diffMins = Math.floor(diffMs / 60000);

  if (diffMins < 1) return 'just now';
  if (diffMins < 60) return `${diffMins}m ago`;

  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;

  const diffDays = Math.floor(diffHours / 24);
  if (diffDays < 7) return `${diffDays}d ago`;

  return dateObj.toLocaleDateString();
}

/**
 * Send message to background script from popup context
 * @param {string} type - Message type
 * @param {Object} data - Message data
 * @description Wraps chrome.runtime.sendMessage for popup
 * @returns {Promise<Object>} Response from background script
 */
function sendToBackgroundPopup(type, data = {}) {
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
 * Escape HTML special characters to prevent injection
 * @param {string} text - Text to escape
 * @description Prevents XSS by escaping HTML entities
 * @returns {string} Escaped HTML-safe text
 */
function escapeHtmlPopup(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// Initialize popup when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initPopup);
} else {
  initPopup();
}

// Cleanup on popup close
window.addEventListener('unload', () => {
  if (healthCheckInterval) {
    clearInterval(healthCheckInterval);
  }
});
