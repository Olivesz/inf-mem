/**
 * inf-mem Background Service Worker
 *
 * Core responsibilities:
 * - Routes messages from content scripts to the FastAPI backend
 * - Manages chrome.storage for settings and state
 * - Provides abstraction for content scripts to access localhost API
 *   (content scripts cannot directly call localhost due to CSP)
 * - Handles all RETRIEVE, EXTRACT, COMPRESS operations
 * - Default settings: { compressionEnabled: false, userId: "default", backendUrl: "http://localhost:7842" }
 */

const DEFAULT_SETTINGS = {
  compressionEnabled: false,
  userId: 'default',
  backendUrl: 'http://localhost:7842'
};

/**
 * Initialize extension on first install/update
 */
chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.sync.get(null, (stored) => {
    if (!stored.backendUrl) {
      chrome.storage.sync.set(DEFAULT_SETTINGS);
    }
  });
});

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
 * Check if backend is accessible
 * @param {Function} sendResponse - callback to send response back
 * @description Makes a HEAD request to backend /health endpoint
 * @returns {Promise<void>} sends {online: boolean} to caller
 */
async function handleHealthCheck(sendResponse) {
  const settings = await getSettings();

  try {
    const response = await fetch(`${settings.backendUrl}/health`, {
      method: 'GET',
      mode: 'no-cors'
    });
    sendResponse({ online: response.ok || response.status === 0 });
  } catch (error) {
    console.error('Health check error:', error);
    sendResponse({ online: false });
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
