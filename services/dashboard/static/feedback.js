/**
 * feedback.js — Feedback review log + notification preferences UI.
 *
 * PURPOSE:
 *   - Displays feedback statistics (accuracy, total verdicts, active rules)
 *   - Shows recent feedback history with inline verdict editing
 *   - Manages suppression rules (toggle, delete)
 *   - Handles notification preference toggles (person/vehicle/suppress known)
 *
 * LOADED BY: index.html before app.js
 * API: /api/feedback, /api/feedback/stats, /api/feedback/rules
 */

// ---------------------------------------------------------------------------
// Feedback Stats + Review
// ---------------------------------------------------------------------------
async function loadFeedbackStats() {
    try {
        const resp = await fetch('/api/feedback/stats');
        if (!resp.ok) return;
        const stats = await resp.json();

        const el = document.getElementById('feedbackStatsContent');
        if (!el) return;

        const total = stats.total_feedback || 0;
        const real = (stats.by_verdict || {}).real_threat || 0;
        const falseA = (stats.by_verdict || {}).false_alarm || 0;
        const identified = (stats.by_verdict || {}).identified || 0;
        const pending = (stats.by_verdict || {}).pending || 0;
        const accuracy = Math.round((stats.alert_accuracy || 0) * 100);
        const rules = stats.active_suppression_rules || 0;

        el.innerHTML = `
            <div class="feedback-stats-grid">
                <div class="feedback-stat">
                    <span class="feedback-stat-value">${total}</span>
                    <span class="feedback-stat-label">Total</span>
                </div>
                <div class="feedback-stat">
                    <span class="feedback-stat-value" style="color: var(--success)">${real}</span>
                    <span class="feedback-stat-label">Real</span>
                </div>
                <div class="feedback-stat">
                    <span class="feedback-stat-value" style="color: var(--warning)">${falseA}</span>
                    <span class="feedback-stat-label">False</span>
                </div>
                <div class="feedback-stat">
                    <span class="feedback-stat-value" style="color: var(--accent)">${accuracy}%</span>
                    <span class="feedback-stat-label">Accuracy</span>
                </div>
                <div class="feedback-stat">
                    <span class="feedback-stat-value" style="color: var(--text-dim)">${rules}</span>
                    <span class="feedback-stat-label">Rules</span>
                </div>
                ${pending > 0 ? `
                <div class="feedback-stat">
                    <span class="feedback-stat-value" style="color: var(--info)">${pending}</span>
                    <span class="feedback-stat-label">Pending</span>
                </div>` : ''}
            </div>
        `;

        // Update badge
        const badge = document.getElementById('feedbackCount');
        if (badge) badge.textContent = total > 0 ? `${total} verdicts` : 'no data';
    } catch (e) {
        console.debug('Feedback stats error:', e);
    }
}


async function loadFeedbackHistory() {
    try {
        const resp = await fetch('/api/feedback?limit=30');
        if (!resp.ok) return;
        const items = await resp.json();

        const el = document.getElementById('feedbackHistoryList');
        if (!el) return;

        if (!items.length) {
            el.innerHTML = '<div class="empty-state">No feedback yet — respond to Telegram alerts to start learning</div>';
            return;
        }

        el.innerHTML = items.map(item => {
            const ts = new Date(item.timestamp * 1000);
            const time = ts.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            const date = ts.toLocaleDateString([], { month: 'short', day: 'numeric' });

            let verdictBadge = '';
            let verdictClass = '';
            switch (item.verdict) {
                case 'real_threat':
                    verdictBadge = '✅ Real';
                    verdictClass = 'verdict-real';
                    break;
                case 'false_alarm':
                    verdictBadge = '❌ False';
                    verdictClass = 'verdict-false';
                    break;
                case 'identified':
                    verdictBadge = '👤 ' + (item.identity_label || 'Unnamed');
                    verdictClass = 'verdict-identified';
                    break;
                case 'pending':
                    verdictBadge = '⏳ Pending';
                    verdictClass = 'verdict-pending';
                    break;
                default:
                    verdictBadge = item.verdict;
            }

            const zone = item.zone ? `<span class="feedback-zone">${item.zone}</span>` : '';
            const eventLabel = item.event_type === 'vehicle_idle' ? '🚗' :
                item.event_type === 'person_identified' ? '👤' : '🚨';

            return `
                <div class="feedback-item">
                    <span class="feedback-event-type">${eventLabel}</span>
                    <span class="feedback-verdict ${verdictClass}">${verdictBadge}</span>
                    ${zone}
                    <span class="feedback-time">${date} ${time}</span>
                </div>
            `;
        }).join('');
    } catch (e) {
        console.debug('Feedback history error:', e);
    }
}


async function loadSuppressionRules() {
    try {
        const resp = await fetch('/api/feedback/rules');
        if (!resp.ok) return;
        const rules = await resp.json();

        const el = document.getElementById('suppressionRulesList');
        if (!el) return;

        if (!rules.length) {
            el.innerHTML = '<div class="empty-state">No suppression rules yet — the system learns from your feedback</div>';
            return;
        }

        el.innerHTML = rules.map(rule => {
            const icon = rule.rule_type === 'identity' ? '👤' : '📍';
            const label = rule.rule_type === 'identity' ?
                `"${rule.identity}"` :
                `${rule.zone} @ ${rule.time_period}`;
            const status = rule.active ? 'Active' : 'Disabled';
            const statusClass = rule.active ? 'rule-active' : 'rule-disabled';

            return `
                <div class="suppression-rule ${statusClass}">
                    <span class="rule-icon">${icon}</span>
                    <span class="rule-label">${label}</span>
                    <span class="rule-meta">${rule.min_false_alarms} false alarms</span>
                    <div class="rule-actions">
                        <button class="rule-toggle" onclick="toggleSuppressionRule(${rule.id}, ${!rule.active})"
                            title="${rule.active ? 'Disable' : 'Enable'}">${rule.active ? '⏸' : '▶'}</button>
                        <button class="rule-delete" onclick="deleteSuppressionRule(${rule.id})"
                            title="Delete rule">🗑️</button>
                    </div>
                </div>
            `;
        }).join('');
    } catch (e) {
        console.debug('Suppression rules error:', e);
    }
}


async function toggleSuppressionRule(ruleId, active) {
    try {
        await fetch(`/api/feedback/rules/${ruleId}/toggle?active=${active}`, { method: 'POST' });
        loadSuppressionRules();
        loadFeedbackStats();
    } catch (e) {
        console.warn('Toggle rule error:', e);
    }
}


async function deleteSuppressionRule(ruleId) {
    if (!confirm('Delete this suppression rule?')) return;
    try {
        await fetch(`/api/feedback/rules/${ruleId}`, { method: 'DELETE' });
        loadSuppressionRules();
        loadFeedbackStats();
    } catch (e) {
        console.warn('Delete rule error:', e);
    }
}


// ---------------------------------------------------------------------------
// Notification Preferences
// ---------------------------------------------------------------------------
async function loadNotificationPrefs() {
    try {
        const resp = await fetch('/api/config');
        if (!resp.ok) return;
        const cfg = await resp.json();

        // Set toggle states from config
        const personToggle = document.getElementById('notifyPersonToggle');
        const vehicleToggle = document.getElementById('notifyVehicleToggle');
        const suppressKnownToggle = document.getElementById('suppressKnownToggle');

        if (personToggle) personToggle.checked = cfg.notify_person === '1';
        if (vehicleToggle) vehicleToggle.checked = cfg.notify_vehicle === '1';
        if (suppressKnownToggle) suppressKnownToggle.checked = cfg.suppress_known === '1';
    } catch (e) {
        console.debug('Load prefs error:', e);
    }
}


function updateNotifPref(key, value) {
    fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [key]: value }),
    }).catch(e => console.warn('Update pref error:', e));
}


// ---------------------------------------------------------------------------
// Init — called by app.js init()
// ---------------------------------------------------------------------------
function initFeedbackPanel() {
    loadFeedbackStats();
    loadFeedbackHistory();
    loadSuppressionRules();
    loadNotificationPrefs();

    // Refresh stats every 30 seconds
    setInterval(() => {
        loadFeedbackStats();
    }, 30000);
}
