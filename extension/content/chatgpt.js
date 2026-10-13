/**
 * inf-mem on chatgpt.com and chat.openai.com: selectors only.
 * The engine is startInfMem() in shared.js.
 *
 * Selectors last written against the live site in 2025 and reproduced in
 * tests/e2e/fixtures/chatgpt.html; that fixture is what CI checks them against.
 */

startInfMem({
  name: 'ChatGPT',
  composer: '#prompt-textarea, div.ProseMirror[contenteditable="true"]',
  sendButton: 'button[data-testid="send-button"]',
  assistantMessages: 'div[data-message-author-role="assistant"]',
  // The send button turns into a stop button while a reply streams.
  isStreaming: () => Boolean(document.querySelector('button[data-testid="stop-button"]')),
  replyText: (el) => el.innerText
});
