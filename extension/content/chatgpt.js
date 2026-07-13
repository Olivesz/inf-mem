/**
 * inf-mem Content Script for ChatGPT
 *
 * Intercepts prompts on chatgpt.com and chat.openai.com:
 * 1. Monitors input field for user prompts via MutationObserver
 * 2. Intercepts send action (Enter key or send button click)
 * 3. Retrieves memories from backend for the prompt
 * 4. Shows memory review panel (memories to inject + compression preview)
 * 5. When user submits, injects memory context into prompt
 * 6. Monitors for assistant response and sends for memory extraction
 *
 * Key selectors for ChatGPT (as of 2025):
 * - Input: div[contenteditable="true"][data-id], div.ProseMirror, div[role="textbox"]
 * - Send button: button[data-testid="send-button"], button[aria-label*="Send"]
 * - Response: div[data-message-author-role="assistant"]
 */

let currentInputElement = null;
let messageHistory = {}; // Track last prompt/response pairs for extraction
let lastPromptSent = null;
let isProcessingIntercept = false;

/**
 * Initialize content script
 * @description Sets up MutationObserver and event listeners
 * @returns {void}
 */
function initChatGPT() {
  console.log('[inf-mem] Initializing ChatGPT content script');

  // Find and monitor the input field
  monitorInputField();

  // Monitor for send button interactions
  observeSendButton();

  // Observe for new assistant messages (for extraction)
  observeResponses();
}

/**
 * Monitor and detect the ChatGPT input field
 * @description Uses MutationObserver to find and attach to the editable input
 * @returns {void}
 */
function monitorInputField() {
  const findInputField = () => {
    const selectors = [
      'div[contenteditable="true"][data-id]',
      'div.ProseMirror',
      'div[role="textbox"]'
    ];

    for (const selector of selectors) {
      const el = document.querySelector(selector);
      if (el) {
        return el;
      }
    }
    return null;
  };

  currentInputElement = findInputField();

  // If not found, set up observer to watch for it
  if (!currentInputElement) {
    const observer = new MutationObserver(() => {
      if (!currentInputElement) {
        currentInputElement = findInputField();
        if (currentInputElement) {
          attachInputListeners();
          observer.disconnect();
        }
      }
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true
    });
  } else {
    attachInputListeners();
  }
}

/**
 * Attach keydown listener to input field
 * @description Intercepts Enter key and redirects to send flow
 * @returns {void}
 */
function attachInputListeners() {
  if (!currentInputElement) return;

  currentInputElement.addEventListener('keydown', async (e) => {
    // Intercept Enter (but not Shift+Enter for newline)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      e.stopPropagation();

      const prompt = currentInputElement.textContent.trim();
      if (prompt) {
        await handlePromptIntercept(prompt);
      }
    }
  });
}

/**
 * Observe send button and attach click listener
 * @description Finds and monitors the send button
 * @returns {void}
 */
function observeSendButton() {
  const findSendButton = () => {
    const selectors = [
      'button[data-testid="send-button"]',
      'button[aria-label*="Send"]'
    ];

    for (const selector of selectors) {
      const el = document.querySelector(selector);
      if (el) {
        return el;
      }
    }
    return null;
  };

  let sendButton = findSendButton();

  const attachSendListener = () => {
    if (!sendButton) return;

    const clickHandler = async (e) => {
      // Prevent default button behavior if we intercept
      const prompt = currentInputElement?.textContent?.trim?.();
      if (prompt && !isProcessingIntercept) {
        e.preventDefault();
        e.stopPropagation();

        await handlePromptIntercept(prompt);
      }
    };

    sendButton.addEventListener('click', clickHandler);
  };

  if (sendButton) {
    attachSendListener();
  }

  // Monitor for send button changes (page reloads, conversation switches, etc.)
  const observer = new MutationObserver(() => {
    const newButton = findSendButton();
    if (newButton && newButton !== sendButton) {
      sendButton = newButton;
      attachSendListener();
    }
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
}

/**
 * Main handler for prompt interception
 * @param {string} prompt - The user's prompt text
 * @description Retrieves memories, shows review panel, handles user choice
 * @returns {Promise<void>}
 */
async function handlePromptIntercept(prompt) {
  if (isProcessingIntercept) return;
  isProcessingIntercept = true;

  try {
    lastPromptSent = prompt;
    const userId = extractUserId();

    // Retrieve memories for this prompt
    console.log('[inf-mem] Retrieving memories for prompt:', prompt.substring(0, 50));
    const retrieveResult = await sendToBackground('RETRIEVE', { prompt, userId });

    if (retrieveResult.error) {
      console.warn('[inf-mem] Retrieval error:', retrieveResult.error);
      // Continue anyway - send prompt without memories
      await sendPrompt(prompt, []);
      isProcessingIntercept = false;
      return;
    }

    const memories = retrieveResult.memories || [];

    // Check if compression is enabled and try to compress
    let compressedPrompt = null;
    try {
      const settings = await sendToBackground('GET_SETTINGS', {});
      if (settings.settings?.compressionEnabled) {
        const compressResult = await sendToBackground('COMPRESS', { prompt });
        compressedPrompt = compressResult.compressed || null;
      }
    } catch (error) {
      console.error('[inf-mem] Compression error:', error);
    }

    // Show review panel
    showPanel(
      memories,
      compressedPrompt,
      prompt,
      () => sendPrompt(prompt, memories),
      () => sendPrompt(prompt, []),
      () => {
        if (compressedPrompt) {
          sendPrompt(compressedPrompt, memories);
        }
      }
    );
  } catch (error) {
    console.error('[inf-mem] Interception error:', error);
    // Fallback: send original prompt
    await sendPrompt(prompt, []);
  } finally {
    isProcessingIntercept = false;
  }
}

/**
 * Send prompt with optional memory context injected
 * @param {string} promptText - The prompt to send (may be original or compressed)
 * @param {Array<Object>} memories - Memories to inject (empty array = no injection)
 * @description Sets input text and triggers native send mechanism
 * @returns {Promise<void>}
 */
async function sendPrompt(promptText, memories) {
  if (!currentInputElement) {
    console.error('[inf-mem] Input element not found');
    return;
  }

  // Inject memory context if memories exist
  const finalPrompt = injectMemoryContext(promptText, memories);

  // Set the input text using React-compatible method
  setInputText(currentInputElement, finalPrompt);

  // Small delay to let React update before triggering send
  await new Promise((resolve) => setTimeout(resolve, 50));

  // Trigger the send action
  triggerSend(currentInputElement);
}

/**
 * Set text in a React-controlled contenteditable input
 * @param {HTMLElement} inputEl - The contenteditable element
 * @param {string} text - Text to set
 * @description Uses execCommand to ensure React detects the change
 * @returns {void}
 */
function setInputText(inputEl, text) {
  // Clear existing content
  inputEl.innerHTML = '';

  // Focus and insert text using execCommand (compatible with React)
  inputEl.focus();
  document.execCommand('insertText', false, text);

  // Dispatch input event for React to detect change
  const inputEvent = new Event('input', { bubbles: true });
  inputEl.dispatchEvent(inputEvent);
}

/**
 * Trigger send action by dispatching Enter key events
 * @param {HTMLElement} inputEl - The input element that has focus
 * @description Simulates pressing Enter to trigger send
 * @returns {void}
 */
function triggerSend(inputEl) {
  const keydownEvent = new KeyboardEvent('keydown', {
    key: 'Enter',
    code: 'Enter',
    keyCode: 13,
    which: 13,
    bubbles: true,
    cancelable: true
  });

  const keyupEvent = new KeyboardEvent('keyup', {
    key: 'Enter',
    code: 'Enter',
    keyCode: 13,
    which: 13,
    bubbles: true,
    cancelable: true
  });

  inputEl.dispatchEvent(keydownEvent);
  inputEl.dispatchEvent(keyupEvent);
}

/**
 * Observe for assistant responses in the conversation
 * @description Monitors for new assistant messages and extracts memories
 * @returns {void}
 */
function observeResponses() {
  const observer = new MutationObserver(async () => {
    const lastMessage = getLastAssistantMessage();

    if (lastMessage && lastPromptSent) {
      const response = lastMessage.textContent.trim();

      // Avoid duplicate extraction
      if (messageHistory[lastPromptSent] !== response) {
        messageHistory[lastPromptSent] = response;

        try {
          const userId = extractUserId();
          console.log('[inf-mem] Extracting memories from response');
          // Fire-and-forget extraction (no UI blocking)
          sendToBackground('EXTRACT', {
            prompt: lastPromptSent,
            response,
            userId
          });
        } catch (error) {
          console.error('[inf-mem] Extraction error:', error);
        }
      }
    }
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true,
    characterData: false
  });
}

/**
 * Get the last assistant message from the conversation
 * @description Finds the most recent assistant message in DOM
 * @returns {HTMLElement|null} The assistant message element or null
 */
function getLastAssistantMessage() {
  const selector = 'div[data-message-author-role="assistant"]';
  const messages = document.querySelectorAll(selector);
  return messages.length > 0 ? messages[messages.length - 1] : null;
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initChatGPT);
} else {
  initChatGPT();
}
