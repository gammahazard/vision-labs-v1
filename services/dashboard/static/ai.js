/**
 * ai.js — AI assistant frontend logic.
 *
 * Handles onboarding wizard state machine, chat message rendering,
 * and API communication with the backend AI routes.
 */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let aiConfig = { enabled: false, user_name: '', ai_name: 'Atlas' };
let chatHistory = [];  // { role: 'user'|'assistant', content: string }
let isWaiting = false;
let ollamaReady = false;

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const resp = await fetch('/api/ai/config');
        if (resp.ok) {
            aiConfig = await resp.json();
        }
    } catch (e) {
        console.warn('Failed to load AI config:', e);
    }

    if (aiConfig.enabled) {
        showChat();
        loadHistory();
        checkOllamaStatus();
    } else {
        showWizard();
    }

    // Chat input — Enter to send, Shift+Enter for newline
    const input = document.getElementById('chatInput');
    if (input) {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        // Auto-resize textarea
        input.addEventListener('input', () => {
            input.style.height = 'auto';
            input.style.height = Math.min(input.scrollHeight, 120) + 'px';
        });
    }
});

// ---------------------------------------------------------------------------
// Wizard
// ---------------------------------------------------------------------------
function showWizard() {
    document.getElementById('wizardContainer').style.display = 'flex';
    document.getElementById('chatContainer').style.display = 'none';
    document.getElementById('wizardStep1').style.display = 'block';
}

function wizardNext(step) {
    // Hide all steps
    for (let i = 1; i <= 3; i++) {
        const el = document.getElementById('wizardStep' + i);
        if (el) el.style.display = 'none';
    }
    // Show target step with animation
    const target = document.getElementById('wizardStep' + step);
    if (target) {
        target.style.display = 'block';
        target.style.animation = 'none';
        // Force reflow
        void target.offsetWidth;
        target.style.animation = 'fadeInUp 0.4s ease-out';
    }
}

function selectAIName(chip, name) {
    // Deselect all
    document.querySelectorAll('.name-chip').forEach(c => c.classList.remove('selected'));
    // Select this one
    chip.classList.add('selected');
    document.getElementById('aiName').value = name;
}

async function finishWizard() {
    const userName = document.getElementById('userName').value.trim();
    const aiName = document.getElementById('aiName').value.trim() || 'Atlas';

    aiConfig = { enabled: true, user_name: userName, ai_name: aiName };

    try {
        const resp = await fetch('/api/ai/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(aiConfig),
        });
        if (resp.ok) {
            aiConfig = await resp.json();
        }
    } catch (e) {
        console.warn('Failed to save AI config:', e);
    }

    showChat();

    // Send introductory message from the AI
    const greeting = userName
        ? `Hey ${userName}! I'm ${aiName}. I'm your local AI assistant — I run entirely on your hardware, so nothing leaves this machine. Ask me about your security cameras, set reminders, or just chat. What can I help you with?`
        : `Hey! I'm ${aiName}. I'm your local AI assistant running right on your hardware. Ask me about security events, set reminders, or just chat. What's on your mind?`;

    addMessage('assistant', greeting);
    checkOllamaStatus();
}

// ---------------------------------------------------------------------------
// Ollama readiness polling
// ---------------------------------------------------------------------------
async function checkOllamaStatus() {
    const overlay = document.getElementById('ollamaOverlay');
    const statusEl = document.getElementById('ollamaStatus');
    const sendBtn = document.getElementById('sendBtn');
    const input = document.getElementById('chatInput');

    // Show overlay immediately
    if (overlay) overlay.style.display = 'flex';
    if (sendBtn) sendBtn.disabled = true;
    if (input) input.placeholder = 'Waiting for AI model...';

    const statusMessages = {
        offline: 'Connecting to Ollama...',
        not_found: 'Model not found — downloading may be in progress...',
        loading: 'Loading Qwen 3 14B into GPU memory...',
        ready: 'Ready!',
    };

    const startTime = Date.now();
    const MAX_WAIT_MS = 120_000; // 2 minutes hard fallback

    const dismissOverlay = () => {
        ollamaReady = true;
        if (sendBtn) sendBtn.disabled = false;
        if (input) input.placeholder = 'Ask me anything...';
        if (overlay) {
            overlay.style.transition = 'opacity 0.5s';
            overlay.style.opacity = '0';
            setTimeout(() => { overlay.style.display = 'none'; }, 500);
        }
        // Re-fetch history to pick up warm-up messages saved after initial load
        loadHistory();
    };

    const poll = async () => {
        // Hard fallback — dismiss after 2 minutes regardless
        if (Date.now() - startTime > MAX_WAIT_MS) {
            console.warn('AI model status check timed out after 2 minutes — dismissing overlay');
            if (statusEl) statusEl.textContent = 'Timed out — try chatting anyway';
            dismissOverlay();
            return;
        }

        try {
            const resp = await fetch('/api/ai/status');
            if (resp.ok) {
                const data = await resp.json();
                if (statusEl) statusEl.textContent = statusMessages[data.status] || data.status;
                if (data.model_ready) {
                    if (statusEl) statusEl.textContent = 'Ready!';
                    dismissOverlay();
                    return; // Stop polling
                }
            }
        } catch (e) {
            if (statusEl) statusEl.textContent = 'Connecting to Ollama...';
        }
        setTimeout(poll, 3000);
    };
    poll();
}

// ---------------------------------------------------------------------------
// Chat UI
// ---------------------------------------------------------------------------
function showChat() {
    document.getElementById('wizardContainer').style.display = 'none';
    document.getElementById('chatContainer').style.display = 'flex';

    // Update header with AI name
    document.getElementById('aiNameDisplay').textContent = aiConfig.ai_name || 'Atlas';

    // Focus input
    setTimeout(() => {
        const input = document.getElementById('chatInput');
        if (input) input.focus();
    }, 100);
}

async function loadHistory() {
    try {
        const resp = await fetch('/api/ai/history?limit=50');
        if (resp.ok) {
            const history = await resp.json();
            if (history.length > 0) {
                chatHistory = history.map(m => ({
                    role: m.role,
                    content: m.content,
                }));
                renderAllMessages();
            } else {
                // No history — show welcome
                const name = aiConfig.ai_name || 'Atlas';
                const greeting = aiConfig.user_name
                    ? `Welcome back, ${aiConfig.user_name}! How can I help?`
                    : `Welcome back! How can I help?`;
                addMessage('assistant', greeting);
            }
        }
    } catch (e) {
        console.warn('Failed to load history:', e);
    }
}

function renderAllMessages() {
    const container = document.getElementById('chatMessages');
    container.innerHTML = '';
    chatHistory.forEach(msg => {
        appendMessageElement(msg.role, msg.content, false);
    });
    scrollToBottom();
}

function addMessage(role, content) {
    chatHistory.push({ role, content });
    appendMessageElement(role, content, true);
    scrollToBottom();
}

function appendMessageElement(role, content, animate) {
    const container = document.getElementById('chatMessages');

    // System messages: centered, muted, no avatar
    if (role === 'system') {
        const sysDiv = document.createElement('div');
        sysDiv.className = 'message-system';
        if (!animate) sysDiv.style.animation = 'none';
        sysDiv.innerHTML = renderMarkdown(content);
        container.appendChild(sysDiv);
        return;
    }

    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;
    if (!animate) msgDiv.style.animation = 'none';

    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    avatarDiv.textContent = role === 'assistant' ? '🤖' : '👤';

    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'message-bubble';
    bubbleDiv.innerHTML = renderMarkdown(content);

    msgDiv.appendChild(avatarDiv);
    msgDiv.appendChild(bubbleDiv);
    container.appendChild(msgDiv);
}

function showTyping() {
    const container = document.getElementById('chatMessages');
    const typing = document.createElement('div');
    typing.className = 'typing-indicator';
    typing.id = 'typingIndicator';

    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    avatarDiv.style.background = 'linear-gradient(135deg, var(--accent), #06b6d4)';
    avatarDiv.textContent = '🤖';

    const dotsDiv = document.createElement('div');
    dotsDiv.className = 'typing-dots';
    dotsDiv.innerHTML = '<span></span><span></span><span></span>';

    typing.appendChild(avatarDiv);
    typing.appendChild(dotsDiv);
    container.appendChild(typing);
    scrollToBottom();
}

function hideTyping() {
    const el = document.getElementById('typingIndicator');
    if (el) el.remove();
}

function scrollToBottom() {
    const container = document.getElementById('chatMessages');
    setTimeout(() => {
        container.scrollTop = container.scrollHeight;
    }, 50);
}

// ---------------------------------------------------------------------------
// Send Message
// ---------------------------------------------------------------------------
async function sendMessage() {
    if (isWaiting) return;
    if (!ollamaReady) {
        showError('AI model is still loading. Please wait...');
        return;
    }

    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    if (!message) return;

    // Add user message
    addMessage('user', message);
    hideSuggestions();
    input.value = '';
    input.style.height = 'auto';

    // Disable input
    isWaiting = true;
    document.getElementById('sendBtn').disabled = true;
    showTyping();
    hideError();

    try {
        const resp = await fetch('/api/ai/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                history: chatHistory.slice(-20),
            }),
        });

        hideTyping();

        if (resp.ok) {
            const data = await resp.json();
            addMessage('assistant', data.reply || 'I had trouble generating a response. Please try again.');
        } else {
            const err = await resp.json().catch(() => ({}));
            showError(err.error || `Error ${resp.status}`);
            // Still add a message so user knows something went wrong
            if (resp.status === 500) {
                addMessage('assistant', '⚠️ I\'m having trouble connecting to my brain (the Qwen model). It might still be downloading — this takes a few minutes on first startup. Try again shortly!');
            }
        }
    } catch (e) {
        hideTyping();
        showError('Network error — is the server running?');
        console.error('Chat error:', e);
    } finally {
        isWaiting = false;
        document.getElementById('sendBtn').disabled = false;
        input.focus();
    }
}

// ---------------------------------------------------------------------------
// Error display
// ---------------------------------------------------------------------------
function showError(msg) {
    const el = document.getElementById('aiError');
    el.textContent = msg;
    el.style.display = 'block';
}

function hideError() {
    document.getElementById('aiError').style.display = 'none';
}

// ---------------------------------------------------------------------------
// Simple Markdown renderer
// ---------------------------------------------------------------------------
function renderMarkdown(text) {
    if (!text) return '';

    // Process line-by-line for proper list handling
    const lines = text.split('\n');
    let html = '';
    let inUl = false, inOl = false;

    for (const line of lines) {
        const trimmed = line.trim();

        // Unordered list item
        if (/^[-*]\s+/.test(trimmed)) {
            if (inOl) { html += '</ol>'; inOl = false; }
            if (!inUl) { html += '<ul>'; inUl = true; }
            html += `<li>${inlineFormat(trimmed.replace(/^[-*]\s+/, ''))}</li>`;
            continue;
        }
        // Ordered list item
        if (/^\d+\.\s+/.test(trimmed)) {
            if (inUl) { html += '</ul>'; inUl = false; }
            if (!inOl) { html += '<ol>'; inOl = true; }
            html += `<li>${inlineFormat(trimmed.replace(/^\d+\.\s+/, ''))}</li>`;
            continue;
        }

        // Close any open list
        if (inUl) { html += '</ul>'; inUl = false; }
        if (inOl) { html += '</ol>'; inOl = false; }

        // Empty line = paragraph break
        if (!trimmed) {
            html += '<br>';
            continue;
        }

        // Image line: ![alt](url)
        if (/^!\[.*?\]\(.*?\)$/.test(trimmed)) {
            if (inUl) { html += '</ul>'; inUl = false; }
            if (inOl) { html += '</ol>'; inOl = false; }
            const match = trimmed.match(/^!\[(.*)\]\((.+)\)$/);
            if (match) {
                html += `<div class="chat-image"><img src="${match[2]}" alt="${match[1]}" style="max-width:100%;border-radius:8px;margin:8px 0;"><div class="chat-image-caption">${match[1]}</div></div>`;
                continue;
            }
        }

        // Video tag pass-through (injected by capture_clip tool)
        if (trimmed.startsWith('<video')) {
            html += trimmed;
            continue;
        }

        // Normal paragraph line
        html += `<p>${inlineFormat(trimmed)}</p>`;
    }

    if (inUl) html += '</ul>';
    if (inOl) html += '</ol>';

    // Clean up double breaks and empty paragraphs
    return html.replace(/<p><\/p>/g, '').replace(/(<br>){3,}/g, '<br><br>');
}

function inlineFormat(text) {
    return text
        // Code blocks (triple backtick — shouldn't appear inline but handle gracefully)
        .replace(/```(\w*)\n?([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
        // Inline code
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        // Inline images: ![alt](url)
        .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" style="max-width:100%;border-radius:8px;margin:4px 0;">')
        // Bold
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        // Italic
        .replace(/\*(.+?)\*/g, '<em>$1</em>');
}

// ---------------------------------------------------------------------------
// Suggestion chips
// ---------------------------------------------------------------------------
function useSuggestion(chip) {
    const text = chip.textContent.replace(/^[\u{1F300}-\u{1FAD6}\u{2600}-\u{27BF}]\s*/u, '').trim();
    const input = document.getElementById('chatInput');
    input.value = text;
    input.focus();
    sendMessage();
}

function hideSuggestions() {
    const el = document.getElementById('suggestionChips');
    if (el) el.style.display = 'none';
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------
async function resetAssistant() {
    if (!confirm('Reset AI assistant? This will clear your chat history and re-open the setup wizard.')) return;
    try {
        await fetch('/api/ai/reset', { method: 'POST' });
    } catch (e) {
        console.warn('Reset failed:', e);
    }
    location.reload();
}
