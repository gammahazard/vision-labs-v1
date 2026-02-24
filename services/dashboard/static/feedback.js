/**
 * feedback.js — Feedback review log + notification preferences UI.
 *
 * PURPOSE:
 *   - Displays feedback statistics (accuracy, total verdicts, active rules)
 *   - Shows recent feedback history with inline verdict editing
 *   - Manages suppression rules (toggle, delete)
 *   - Handles notification preference toggles (person/vehicle/suppress known)
 *   - Provides naming modal for unnamed identifications and pending events
 *
 * LOADED BY: index.html before app.js
 * API: /api/feedback, /api/feedback/stats, /api/feedback/rules
 */

// Track which event we're naming
let _feedbackNamingEventId = null;

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
        const real = (stats.by_verdict || {}).real_detection || 0;
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
            let actions = '';
            switch (item.verdict) {
                case 'real_detection':
                    verdictBadge = '✅ Real';
                    verdictClass = 'verdict-real';
                    break;
                case 'false_alarm':
                    verdictBadge = '❌ False';
                    verdictClass = 'verdict-false';
                    break;
                case 'identified':
                    if (!item.identity_label) {
                        verdictBadge = '👤 Unnamed';
                        verdictClass = 'verdict-identified feedback-clickable';
                    } else {
                        verdictBadge = '👤 ' + item.identity_label;
                        verdictClass = 'verdict-identified';
                    }
                    break;
                case 'pending':
                    verdictBadge = '⏳ Pending';
                    verdictClass = 'verdict-pending';
                    actions = `
                        <span class="feedback-actions">
                            <button class="feedback-action-btn fb-real" onclick="event.stopPropagation(); quickResolve('${item.event_id}', 'real_detection')" title="Real detection">✅</button>
                            <button class="feedback-action-btn fb-false" onclick="event.stopPropagation(); quickResolve('${item.event_id}', 'false_alarm')" title="False alarm">❌</button>
                            <button class="feedback-action-btn fb-name" onclick="event.stopPropagation(); openFeedbackNameModal('${item.event_id}')" title="Name this person">👤</button>
                        </span>`;
                    break;
                default:
                    verdictBadge = item.verdict;
            }

            const zone = item.zone ? `<span class="feedback-zone">${item.zone}</span>` : '';
            const eventLabel = item.event_type === 'vehicle_idle' ? '🚗' :
                item.event_type === 'person_identified' ? '👤' : '🚨';

            // All items are clickable — open event detail modal with snapshot + edit
            const clickHandler = `onclick="_openEventDetail({
                eventId: '${item.event_id.replace(/'/g, "\\'")}',
                eventTitle: '${(item.event_type || 'Event').replace(/_/g, ' ')}',
                eventMeta: '${date} ${time}${item.zone ? ' · ' + item.zone : ''}',
                identityName: '${(item.identity_label || '').replace(/'/g, "\\'")}',
                zone: '${(item.zone || '').replace(/'/g, "\\'")}',
                action: '${(item.action || '').replace(/'/g, "\\'")}'
            })"`;

            return `
                <div class="feedback-item" ${clickHandler} style="cursor:pointer" title="Click to view snapshot & edit">
                    <span class="feedback-event-type">${eventLabel}</span>
                    <span class="feedback-verdict ${verdictClass}">${verdictBadge}</span>
                    ${zone}
                    ${actions}
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
        const data = await resp.json();
        const cfg = data.config || {};

        // Set toggle states from config
        const personToggle = document.getElementById('notifyPersonToggle');
        const vehicleToggle = document.getElementById('notifyVehicleToggle');
        const suppressKnownToggle = document.getElementById('suppressKnownToggle');

        if (personToggle) personToggle.checked = cfg.notify_person !== '0';
        if (vehicleToggle) vehicleToggle.checked = cfg.notify_vehicle !== '0';
        if (suppressKnownToggle) suppressKnownToggle.checked = cfg.suppress_known === '1';

        // Set cooldown slider values from config
        const notifyCooldown = document.getElementById('notifyCooldownSlider');
        const vehicleCooldown = document.getElementById('vehicleCooldownSlider');
        if (notifyCooldown && cfg.notify_cooldown) {
            notifyCooldown.value = cfg.notify_cooldown;
            document.getElementById('notifyCooldownValue').textContent = cfg.notify_cooldown + 's';
        }
        if (vehicleCooldown && cfg.vehicle_cooldown) {
            vehicleCooldown.value = cfg.vehicle_cooldown;
            document.getElementById('vehicleCooldownValue').textContent = cfg.vehicle_cooldown + 's';
        }
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

    // Sync notification toggles every 10s (picks up Telegram arm/disarm)
    setInterval(loadNotificationPrefs, 10000);
}


// ---------------------------------------------------------------------------
// Feedback Naming Modal — name unnamed identifications or pending events
// ---------------------------------------------------------------------------
function openFeedbackNameModal(eventId) {
    _feedbackNamingEventId = eventId;
    const modal = document.getElementById('feedbackNameModal');
    const input = document.getElementById('feedbackNameInput');
    const preview = document.getElementById('feedbackSnapshotPreview');
    if (modal) {
        modal.style.display = 'flex';
        input.value = '';
        // Load event snapshot
        if (preview) {
            preview.style.display = 'none';
            preview.src = '';
            const snapshotUrl = `/api/events/${encodeURIComponent(eventId)}/snapshot`;
            const img = new Image();
            img.onload = () => {
                preview.src = snapshotUrl;
                preview.style.display = 'block';
            };
            img.onerror = () => { preview.style.display = 'none'; };
            img.src = snapshotUrl;
        }
        setTimeout(() => input.focus(), 100);
    }
}

function closeFeedbackNameModal() {
    _feedbackNamingEventId = null;
    const modal = document.getElementById('feedbackNameModal');
    const preview = document.getElementById('feedbackSnapshotPreview');
    if (modal) modal.style.display = 'none';
    if (preview) { preview.src = ''; preview.style.display = 'none'; }
}

async function submitFeedbackName() {
    const name = (document.getElementById('feedbackNameInput')?.value || '').trim();
    if (!name || !_feedbackNamingEventId) return;

    try {
        const resp = await fetch(`/api/feedback/${_feedbackNamingEventId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ verdict: 'identified', identity_label: name }),
        });
        if (resp.ok) {
            closeFeedbackNameModal();
            loadFeedbackHistory();
            loadFeedbackStats();
        } else {
            console.warn('Failed to submit name:', await resp.text());
        }
    } catch (e) {
        console.warn('Submit feedback name error:', e);
    }
}

/**
 * Quick-resolve a pending event directly from the dashboard (✅/❌ buttons).
 */
async function quickResolve(eventId, verdict) {
    try {
        const resp = await fetch(`/api/feedback/${eventId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ verdict }),
        });
        if (resp.ok) {
            loadFeedbackHistory();
            loadFeedbackStats();
        }
    } catch (e) {
        console.warn('Quick resolve error:', e);
    }
}

// Close naming modal on Enter key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && _feedbackNamingEventId) {
        submitFeedbackName();
    } else if (e.key === 'Escape' && _feedbackNamingEventId) {
        closeFeedbackNameModal();
    }
});
