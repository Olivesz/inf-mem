/**
 * inf-mem Background Service Worker
 *
 * Core responsibilities:
 * - Routes messages from content scripts to the FastAPI backend
 * - Manages chrome.storage for settings and state
 * - Provides abstraction for content scripts to access localhost API
 *   (content scripts cannot directly call localhost due to CSP)
 * - Handles RETRIEVE, EXTRACT, COMPRESS and the popup's memory list,
 *   delete and clear-all
 * - Default settings: { compressionEnabled: false, userId: "default", backendUrl: "http://localhost:7842" },
 *   merged in on every read by getSettings(), never written over stored values
 */

const DEFAULT_SETTINGS = {
  compressionEnabled: false,
  userId: 'default',
  backendUrl: 'http://localhost:7842'
};

/**
 * Message router - handles all communication from content scripts
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  (async () => {
    try {
      const { type, ...data } = message;

      switch (type) {
        case 'RETRIEVE':
          await handleRetrieve(data, sendResponse);
          break;
        case 'EXTRACT':
          await handleExtract(data, sendResponse);
          break;
        case 'COMPRESS':
          await handleCompress(data, sendResponse);
          break;
        case 'GET_SETTINGS':
          await handleGetSettings(sendResponse);
          break;
        case 'SAVE_SETTINGS':
          await handleSaveSettings(data, sendResponse);
          break;
        case 'HEALTH_CHECK':
          await handleHealthCheck(sendResponse);
          break;
        case 'GET_MEMORIES':
          await handleGetMemories(sendResponse);
          break;
        case 'DELETE_MEMORY':
          await handleDeleteMemory(data, sendResponse);
          break;
        case 'CLEAR_MEMORIES':
          await handleClearMemories(sendResponse);
          break;
        default:
          sendResponse({ error: `Unknown message type: ${type}` });
      }
    } catch (error) {
      console.error('Background script error:', error);
      sendResponse({ error: error.message });
    }
  })();

  // Return true to indicate we'll send response asynchronously
  return true;
});

/**
 * Retrieve memories for a given prompt
 * @param {Object} data - { prompt: string, userId: string }
 * @param {Function} sendResponse - callback to send response back
 * @description Calls POST /retrieve on backend with prompt and userId
 * @returns {Promise<void>} sends {memories: Array, error?: string} to caller
 */
async function handleRetrieve(data, sendResponse) {
  const settings = await getSettings();
  const { prompt, userId = settings.userId } = data;

  try {
    const response = await fetch(`${settings.backendUrl}/retrieve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, userId })
    });

    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}: ${response.statusText}`);
    }

    const result = await response.json();
    sendResponse({ memories: result.memories || [] });
  } catch (error) {
    console.error('RETRIEVE error:', error);
    sendResponse({
      error: 'Backend offline. Start with: make start',
      memories: []
    });
  }
}

/**
 * Extract memories from prompt/response pair
 * @param {Object} data - { prompt: string, response: string, userId: string }
 * @param {Function} sendResponse - callback to send response back
 * @description Calls POST /extract on backend; fire-and-forget (doesn't block UI)
 * @returns {Promise<void>} sends {success: boolean, error?: string} to caller
 */
async function handleExtract(data, sendResponse) {
  const settings = await getSettings();
  const { prompt, response, userId = settings.userId } = data;

  // Fire-and-forget: don't await this call
  fetch(`${settings.backendUrl}/extract`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, response, userId })
  }).catch((error) => {
    console.error('EXTRACT error:', error);
  });

  sendResponse({ success: true });
}

/**
 * Compress a prompt using backend
 * @param {Object} data - { prompt: string }
 * @param {Function} sendResponse - callback to send response back
 * @description Calls POST /compress on backend with prompt
 * @returns {Promise<void>} sends {compressed: string, error?: string} to caller
 */
async function handleCompress(data, sendResponse) {
  const settings = await getSettings();
  const { prompt } = data;

  try {
    const response = await fetch(`${settings.backendUrl}/compress`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt })
    });

    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}: ${response.statusText}`);
    }

    const result = await response.json();
    sendResponse({ compressed: result.compressed || prompt });
  } catch (error) {
    console.error('COMPRESS error:', error);
    sendResponse({
      error: 'Backend offline. Start with: make start',
      compressed: null
    });
  }
}

/**
 * Get current extension settings from chrome.storage
 * @param {Function} sendResponse - callback to send response back
 * @description Retrieves stored settings or returns defaults
 * @returns {Promise<void>} sends {settings: Object} to caller
 */
async function handleGetSettings(sendResponse) {
  const settings = await getSettings();
  sendResponse({ settings });
}

/**
 * Save extension settings to chrome.storage
 * @param {Object} data - { settings: Object }
 * @param {Function} sendResponse - callback to send response back
 * @description Merges provided settings with existing ones
 * @returns {Promise<void>} sends {success: boolean} to caller
 */
async function handleSaveSettings(data, sendResponse) {
  const { settings } = data;
  await new Promise((resolve) => {
    chrome.storage.sync.set(settings, resolve);
  });
  sendResponse({ success: true });
}

/**
 * Check the backend and report what /health says
 * @param {Function} sendResponse - callback to send response back
 * @returns {Promise<void>} sends {online, memoryCount, groqReachable} to caller
 */
async function handleHealthCheck(sendResponse) {
  const settings = await getSettings();

  try {
    const response = await fetch(`${settings.backendUrl}/health`);
    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }
    const health = await response.json();
    sendResponse({
      online: true,
      memoryCount: health.memory_count,
      groqReachable: health.groq_reachable
    });
  } catch (error) {
    console.error('Health check error:', error);
    sendResponse({ online: false });
  }
}

/**
 * List the user's memories from the backend, newest first
 * @param {Function} sendResponse - callback to send response back
 * @returns {Promise<void>} sends {memories: Array, error?: string} to caller
 */
async function handleGetMemories(sendResponse) {
  const settings = await getSettings();
  const url = `${settings.backendUrl}/memories?user_id=${encodeURIComponent(settings.userId)}`;

  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }
    const result = await response.json();
    sendResponse({ memories: result.memories || [] });
  } catch (error) {
    console.error('GET_MEMORIES error:', error);
    sendResponse({ error: 'Backend offline. Start with: make start', memories: [] });
  }
}

/**
 * Delete one memory on the backend
 * @param {Object} data - { id: string }
 * @param {Function} sendResponse - callback to send response back
 * @returns {Promise<void>} sends {success: boolean, error?: string} to caller
 */
async function handleDeleteMemory(data, sendResponse) {
  const settings = await getSettings();

  try {
    const response = await fetch(`${settings.backendUrl}/memories/${encodeURIComponent(data.id)}`, {
      method: 'DELETE'
    });
    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }
    sendResponse({ success: true });
  } catch (error) {
    console.error('DELETE_MEMORY error:', error);
    sendResponse({ success: false, error: error.message });
  }
}

/**
 * Delete all of the current user's memories on the backend
 * @param {Function} sendResponse - callback to send response back
 * @returns {Promise<void>} sends {success: boolean, deleted?: number, error?: string} to caller
 */
async function handleClearMemories(sendResponse) {
  const settings = await getSettings();
  const url = `${settings.backendUrl}/memories?user_id=${encodeURIComponent(settings.userId)}`;

  try {
    const response = await fetch(url, { method: 'DELETE' });
    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }
    const result = await response.json();
    sendResponse({ success: true, deleted: result.deleted });
  } catch (error) {
    console.error('CLEAR_MEMORIES error:', error);
    sendResponse({ success: false, error: error.message });
  }
}

/**
 * Helper: Get settings from chrome.storage with defaults
 * @description Merges stored settings with defaults
 * @returns {Promise<Object>} merged settings object
 */
async function getSettings() {
  return new Promise((resolve) => {
    chrome.storage.sync.get(DEFAULT_SETTINGS, (stored) => {
      resolve({ ...DEFAULT_SETTINGS, ...stored });
    });
  });
}
