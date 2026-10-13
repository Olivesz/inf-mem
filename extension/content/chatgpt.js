/**
 * inf-mem on chatgpt.com and chat.openai.com: selectors only.
 * The engine is startInfMem() in shared.js.
 *
 * Two composer generations are matched. The textarea + data-composer-submit
 * form was captured from the live site on 2026-10-13 (logged out, desktop
 * width; docs/live-checks/2026-10-13-chatgpt.json) and is reproduced in
 * tests/e2e/fixtures/chatgpt-live.html. The ProseMirror + data-testid markup
 * is the earlier logged-in layout, kept in tests/e2e/fixtures/chatgpt.html.
 */

startInfMem({
  name: 'ChatGPT',
  composer: 'textarea[name="prompt"], #prompt-textarea, div.ProseMirror[contenteditable="true"]',
  sendButton: 'button[data-composer-submit], button[data-testid="send-button"]',
  assistantMessages: 'div[data-message-author-role="assistant"]',
  // While a reply streams, the send button becomes a stop button.
  isStreaming: () => Boolean(document.querySelector(
    'button[data-testid="stop-button"], button[data-composer-submit][aria-label="Stop generating"]'
  )),
  replyText: (el) => el.innerText
});
