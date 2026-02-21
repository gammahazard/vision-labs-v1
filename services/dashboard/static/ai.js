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

    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    if (!message) return;

    // Add user message
    addMessage('user', message);
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

    return text
        // Code blocks
        .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
        // Inline code
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        // Bold
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        // Italic
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        // Line breaks — double newline = paragraph
        .replace(/\n\n/g, '</p><p>')
        // Single newline = br
        .replace(/\n/g, '<br>')
        // Wrap in paragraph
        .replace(/^(.+)$/s, '<p>$1</p>')
        // Unordered lists
        .replace(/<br>- /g, '</p><ul><li>')
        .replace(/<li>([^<]*)/g, '<li>$1</li>')
        // Clean up
        .replace(/<\/li><\/p>/g, '</li></ul>')
        .replace(/<p><\/p>/g, '');
}
