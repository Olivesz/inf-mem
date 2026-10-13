/**
 * inf-mem on claude.ai: selectors only.
 * The engine is startInfMem() in shared.js.
 *
 * Selectors last written against the live site in 2025 and reproduced in
 * tests/e2e/fixtures/claude.html; that fixture is what CI checks them against.
 */

startInfMem({
  name: 'Claude',
  composer: 'div.ProseMirror[contenteditable="true"]',
  sendButton: 'button[aria-label="Send message"], button[aria-label="Send Message"]',
  // Each reply is wrapped in an element carrying data-is-streaming.
  assistantMessages: 'div[data-is-streaming]',
  isStreaming: () => Boolean(document.querySelector('div[data-is-streaming="true"]')),
  replyText: (el) => (el.querySelector('.font-claude-message') || el).innerText
});
