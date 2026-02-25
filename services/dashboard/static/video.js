/**
 * video.js — Video Pipeline tab logic
 *
 * Handles:
 * - Script generation via Ollama (POST /api/video/script)
 * - Full pipeline execution (POST /api/video/generate)
 * - Progress polling (GET /api/video/status/{job_id})
 * - Pipeline cancellation
 * - Video listing & playback
 */
(function () {
    'use strict';

    const $ = id => document.getElementById(id);

    let currentScript = null;
    let currentJobId = null;
    let pollTimer = null;
    let sceneImages = {};  // scene_number -> { filename, url }

    // --- Load models into video model dropdown (reuse image gen endpoint) ---
    async function loadVideoModels() {
        try {
            const resp = await fetch('/api/generate/models');
            const data = await resp.json();
            const sel = $('videoModel');
            if (!sel || !data.models) return;
            sel.innerHTML = '';
            data.models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m;
                opt.textContent = m.replace(/\.[^.]+$/, '');
                sel.appendChild(opt);
            });
        } catch (e) {
            console.warn('Failed to load video models:', e);
        }
    }

    // --- Script Generation ---
    window._videoGenerateScript = async function () {
        const prompt = $('videoPrompt').value.trim();
        if (!prompt) return;

        const btn = $('videoScriptBtn');
        btn.disabled = true;
        btn.textContent = '⏳ Generating script...';
        $('videoStatusText').textContent = 'Generating script...';

        try {
            const duration = $('videoDuration').value;
            const resp = await fetch('/api/video/script', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt,
                    target_minutes: parseInt(duration),
                }),
            });

            const data = await resp.json();

            if ('error' in data) {
                $('videoStatusText').textContent = `Error: ${data.error || 'Unknown error from server'}`;
                btn.disabled = false;
                btn.textContent = '📝 Generate Script';
                return;
            }

            if (!data.script || !data.script.scenes) {
                $('videoStatusText').textContent = `Error: Invalid response — no script returned. Raw: ${JSON.stringify(data).substring(0, 200)}`;
                btn.disabled = false;
                btn.textContent = '📝 Generate Script';
                return;
            }

            currentScript = data.script;
            renderScript(data);

            // Persist script to localStorage so it survives refresh/rebuild
            try { localStorage.setItem('video_script', JSON.stringify(data)); } catch (e) { }
            // Also persist sceneImages
            try { localStorage.setItem('video_scene_images', JSON.stringify(sceneImages)); } catch (e) { }

            $('videoProduceBtn').disabled = false;
            $('videoStatusText').textContent = `Script ready — ${data.scene_count} scenes, ~${data.estimated_minutes} min`;

        } catch (e) {
            $('videoStatusText').textContent = `Error: ${e.message}`;
        }

        btn.disabled = false;
        btn.textContent = '📝 Generate Script';
    };

    // --- Render Script ---
    function renderScript(data) {
        const editor = $('videoScriptEditor');
        const title = $('videoScriptTitle');
        const info = $('videoScriptInfo');
        const list = $('videoScenesList');

        editor.style.display = 'block';
        title.textContent = data.script.title || 'Untitled Script';
        info.textContent = `${data.scene_count} scenes · ~${data.estimated_minutes} min`;

        sceneImages = {};  // Reset scene images

        let html = '';
        data.script.scenes.forEach((scene, idx) => {
            const sn = scene.scene_number || scene.number || (idx + 1);
            scene.scene_number = sn;  // normalize for later use
            const narration = scene.narration || scene.dialogue || scene.text || '';
            const prompt = scene.image_prompt || scene.visual || scene.prompt || '';
            const dur = scene.duration_seconds || scene.duration || 5;
            html += `
                <div class="video-scene-card" id="scene-card-${sn}">
                    <div class="video-scene-number">${sn}</div>
                    <div class="video-scene-content">
                        <div class="video-scene-narration">"${narration}"</div>
                        <div class="video-scene-prompt">${prompt}</div>
                        <div class="video-scene-duration">${dur}s</div>
                        <div class="video-scene-image-row">
                            <div class="video-scene-thumb" id="scene-thumb-${sn}" style="display:none"></div>
                            <button class="btn-scene-image" onclick="window._videoPickImage(${sn})" title="Attach image for i2v animation">
                                📎 Image
                            </button>
                            <button class="btn-scene-image-clear" id="scene-clear-${sn}" onclick="window._videoClearImage(${sn})" style="display:none" title="Remove image">
                                ✕
                            </button>
                        </div>
                    </div>
                </div>
            `;
        });
        list.innerHTML = html;

        // Show characters if present
        if (data.script.characters && data.script.characters.length > 0) {
            let charHtml = '<div class="video-scene-card" style="border-color: rgba(139,92,246,0.2)">';
            charHtml += '<div class="video-scene-number">👤</div>';
            charHtml += '<div class="video-scene-content">';
            data.script.characters.forEach(c => {
                charHtml += `<div class="video-scene-narration" style="font-style:normal"><strong>${c.name}:</strong> ${c.description}</div>`;
            });
            charHtml += '</div></div>';
            list.insertAdjacentHTML('afterbegin', charHtml);
        }
    }

    // --- Produce Full Video ---
    window._videoProduceFull = async function () {
        if (!currentScript) return;

        const btn = $('videoProduceBtn');
        btn.disabled = true;
        btn.textContent = '⏳ Starting pipeline...';

        const res = $('videoResolution').value.split('x');

        try {
            const resp = await fetch('/api/video/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    script: {
                        ...currentScript,
                        scenes: currentScript.scenes.map(s => ({
                            ...s,
                            source_image: sceneImages[s.scene_number]?.filename || '',
                        })),
                    },
                    settings: {
                        model: $('videoModel').value || 'zillah.safetensors',
                        width: parseInt(res[0]),
                        height: parseInt(res[1]),
                        steps: 20,
                        cfg: 7.0,
                        smooth: $('videoSmooth')?.checked || false,
                    },
                }),
            });

            const data = await resp.json();

            if (data.error) {
                $('videoStatusText').textContent = `Error: ${data.error}`;
                btn.disabled = false;
                btn.textContent = '🎬 Produce Video';
                return;
            }

            currentJobId = data.job_id;
            $('videoProgress').style.display = 'block';
            $('videoStatusText').textContent = `Pipeline running — Job ${data.job_id}`;

            // Start polling
            startPolling(data.job_id);

        } catch (e) {
            $('videoStatusText').textContent = `Error: ${e.message}`;
            btn.disabled = false;
            btn.textContent = '🎬 Produce Video';
        }
    };

    // --- Progress Polling ---
    let _videoPolling = false;  // reentrance guard

    function startPolling(jobId) {
        if (pollTimer) clearInterval(pollTimer);
        _videoPolling = false;

        pollTimer = setInterval(async () => {
            if (_videoPolling) return;  // skip if previous poll still in-flight
            _videoPolling = true;

            try {
                const resp = await fetch(`/api/video/status/${jobId}`);
                const job = await resp.json();

                if (job.error) {
                    stopPolling();
                    return;
                }

                // Update progress
                $('videoProgressBar').style.width = `${job.progress}%`;
                $('videoProgressText').textContent = `${job.progress}% — ${job.message || ''}`;
                $('videoProgressPhase').textContent = `Phase: ${job.phase || '?'}`;
                $('videoStatusText').textContent = job.message || 'Working...';

                if (job.status === 'complete') {
                    stopPolling();
                    $('videoStatusText').textContent = `✅ Video complete!`;
                    $('videoProduceBtn').disabled = false;
                    $('videoProduceBtn').textContent = '🎬 Produce Video';

                    // Show video player
                    if (job.output_file) {
                        $('videoOutput').innerHTML = `
                            <div style="width:100%; text-align:center;">
                                <video controls style="max-width:100%; border-radius:12px; box-shadow: 0 8px 32px rgba(0,0,0,0.4);">
                                    <source src="/api/video/download/${encodeURIComponent(job.output_file)}" type="video/mp4">
                                </video>
                                <p style="color:#71717a; margin-top:8px; font-size:0.85em;">${job.output_file}</p>
                            </div>
                        `;
                    }

                    loadVideoHistory();

                } else if (job.status === 'error' || job.status === 'cancelled') {
                    stopPolling();
                    $('videoStatusText').textContent = `❌ ${job.message}`;
                    $('videoProduceBtn').disabled = false;
                    $('videoProduceBtn').textContent = '🎬 Produce Video';
                }
            } catch (e) {
                console.warn('Poll error:', e);
            } finally {
                _videoPolling = false;
            }
        }, 2000);
    }

    function stopPolling() {
        if (pollTimer) {
            clearInterval(pollTimer);
            pollTimer = null;
        }
    }

    // --- Cancel Pipeline ---
    window._videoCancelPipeline = async function () {
        if (!currentJobId) return;
        try {
            await fetch(`/api/video/cancel/${currentJobId}`, { method: 'POST' });
            $('videoStatusText').textContent = 'Cancelling...';
        } catch (e) {
            console.warn('Cancel error:', e);
        }
    };

    // --- Clear All / Reset ---
    window._videoClearAll = function () {
        if (!confirm('Clear the script, scene images, and start fresh?')) return;

        // Reset state
        currentScript = null;
        currentJobId = null;
        sceneImages = {};
        stopPolling();

        // Clear localStorage
        localStorage.removeItem('video_script');
        localStorage.removeItem('video_scene_images');

        // Clear UI
        $('videoPrompt').value = '';
        $('videoScriptEditor').style.display = 'none';
        $('videoScenesList').innerHTML = '';
        $('videoProduceBtn').disabled = true;
        $('videoStatusText').textContent = 'Ready';
        $('videoStatusDot').className = 'video-status-dot';

        // Hide progress if visible
        var progress = $('videoProgress');
        if (progress) progress.style.display = 'none';
    };

    // --- Video History ---
    async function loadVideoHistory() {
        try {
            const resp = await fetch('/api/video/list');
            const data = await resp.json();

            if (!data.videos || data.videos.length === 0) {
                $('videoHistory').style.display = 'none';
                return;
            }

            $('videoHistory').style.display = 'block';
            let html = '';
            data.videos.forEach(v => {
                html += `
                    <div class="video-history-item">
                        <div class="video-history-info" onclick="window._videoPlay('${v.filename}')">
                            <span class="video-history-name">🎬 ${v.filename}</span>
                            <span class="video-history-meta">${v.size_mb} MB</span>
                        </div>
                        <button class="video-history-delete" onclick="event.stopPropagation(); window._videoDelete('${v.filename}')" title="Delete video">✕</button>
                    </div>
                `;
            });
            $('videoHistoryList').innerHTML = html;

        } catch (e) {
            console.warn('Video list error:', e);
        }
    }

    window._videoDelete = async function (filename) {
        if (!confirm('Delete video "' + filename + '"?')) return;
        try {
            await fetch('/api/video/' + encodeURIComponent(filename), { method: 'DELETE' });
            loadVideoHistory();
            var output = $('videoOutput');
            if (output && output.innerHTML.includes(filename)) {
                output.innerHTML = '<div class="video-output-placeholder"><span class="placeholder-icon">🎬</span><span>Your video will appear here</span></div>';
            }
        } catch (e) {
            console.warn('Delete failed:', e);
        }
    };

    window._videoPlay = function (filename) {
        $('videoOutput').innerHTML = `
            <div style="width:100%; text-align:center; position:relative;">
                <button class="video-close-btn" onclick="window._videoClosePlayer()" title="Close video">✕</button>
                <video controls autoplay style="max-width:100%; border-radius:12px; box-shadow: 0 8px 32px rgba(0,0,0,0.4);">
                    <source src="/api/video/download/${encodeURIComponent(filename)}" type="video/mp4">
                </video>
                <p style="color:#71717a; margin-top:8px; font-size:0.85em;">${filename}</p>
            </div>
        `;
    };

    window._videoClosePlayer = function () {
        $('videoOutput').innerHTML = '<div class="video-output-placeholder"><span class="placeholder-icon">🎬</span><span>Your video will appear here</span></div>';
    };

    // --- Init Video Tab ---
    window.initVideoTab = function () {
        loadVideoModels();
        loadVideoHistory();
        loadCharacters();

        // Restore last script from localStorage
        try {
            const saved = localStorage.getItem('video_script');
            if (saved) {
                const data = JSON.parse(saved);
                if (data.script && data.script.scenes && data.script.scenes.length > 0) {
                    currentScript = data.script;
                    renderScript(data);
                    $('videoProduceBtn').disabled = false;
                    $('videoStatusText').textContent = `Restored script — ${data.scene_count} scenes, ~${data.estimated_minutes} min`;

                    // Restore scene images
                    try {
                        const savedImages = localStorage.getItem('video_scene_images');
                        if (savedImages) {
                            sceneImages = JSON.parse(savedImages);
                            for (const [sn, img] of Object.entries(sceneImages)) {
                                const thumbEl = document.getElementById(`scene-thumb-${sn}`);
                                const clearBtn = document.getElementById(`scene-clear-${sn}`);
                                if (thumbEl && img.url) {
                                    thumbEl.innerHTML = `<img src="${img.url}" alt="Scene ${sn}">`;
                                    thumbEl.style.display = 'block';
                                } else if (thumbEl && img.filename) {
                                    thumbEl.innerHTML = `<span style="color:#a78bfa">📎 ${img.filename}</span>`;
                                    thumbEl.style.display = 'block';
                                }
                                if (clearBtn) clearBtn.style.display = 'inline-block';
                            }
                        }
                    } catch (e) {
                        console.warn('Failed to restore scene images:', e);
                    }
                }
            }
        } catch (e) {
            console.warn('Failed to restore video script:', e);
        }
    };

    // --- Scene Image Picker ---
    window._videoPickImage = async function (sceneNum) {
        // Create a modal to pick from gallery or upload
        let overlay = document.getElementById('videoImagePickerOverlay');
        if (overlay) overlay.remove();

        overlay = document.createElement('div');
        overlay.id = 'videoImagePickerOverlay';
        overlay.className = 'video-image-picker-overlay';
        overlay.innerHTML = `
            <div class="video-image-picker-modal">
                <div class="video-image-picker-header">
                    <h3>📎 Attach Image — Scene ${sceneNum}</h3>
                    <button onclick="this.closest('.video-image-picker-overlay').remove()" class="btn-close">✕</button>
                </div>
                <div class="video-image-picker-upload">
                    <label class="btn-upload-scene-img">
                        📁 Upload from Computer
                        <input type="file" accept="image/*" onchange="window._videoUploadSceneImage(${sceneNum}, this.files[0])" style="display:none">
                    </label>
                </div>
                <div class="video-image-picker-gallery" id="videoImagePickerGallery">
                    <p style="color:#71717a">Loading gallery...</p>
                </div>
            </div>
        `;
        document.body.appendChild(overlay);
        overlay.addEventListener('click', e => {
            if (e.target === overlay) overlay.remove();
        });

        // Load gallery images
        try {
            const resp = await fetch('/api/video/gallery-images');
            const data = await resp.json();
            const gallery = document.getElementById('videoImagePickerGallery');
            if (!data.images || data.images.length === 0) {
                gallery.innerHTML = '<p style="color:#71717a">No gallery images yet — generate some in the Generate tab!</p>';
                return;
            }
            let html = '';
            data.images.forEach(img => {
                html += `<img src="${img.url}" class="video-gallery-thumb" 
                    onclick="window._videoSelectGalleryImage(${sceneNum}, '${img.filename}', '${img.url}')" 
                    title="${img.filename}">`;
            });
            gallery.innerHTML = html;
        } catch (e) {
            document.getElementById('videoImagePickerGallery').innerHTML = `<p style="color:#ef4444">${e.message}</p>`;
        }
    };

    window._videoSelectGalleryImage = async function (sceneNum, filename, url) {
        // Upload this gallery image to ComfyUI's input folder
        const thumbEl = document.getElementById(`scene-thumb-${sceneNum}`);
        thumbEl.innerHTML = '<span style="color:#a78bfa">Uploading...</span>';
        thumbEl.style.display = 'block';

        try {
            const imgResp = await fetch(url);
            const blob = await imgResp.blob();
            const formData = new FormData();
            formData.append('file', blob, filename);

            const resp = await fetch('/api/video/scene-image', {
                method: 'POST',
                body: formData,
            });
            const data = await resp.json();
            if (data.error) throw new Error(data.error);

            sceneImages[sceneNum] = { filename: data.filename, url };
            try { localStorage.setItem('video_scene_images', JSON.stringify(sceneImages)); } catch (e) { }
            thumbEl.innerHTML = `<img src="${url}" alt="Scene ${sceneNum}">`;
            document.getElementById(`scene-clear-${sceneNum}`).style.display = 'inline-block';
            // Update button text
            const card = document.getElementById(`scene-card-${sceneNum}`);
            const btn = card.querySelector('.btn-scene-image');
            btn.textContent = '🔄 Change';
        } catch (e) {
            thumbEl.innerHTML = `<span style="color:#ef4444">${e.message}</span>`;
        }

        // Close picker
        const overlay = document.getElementById('videoImagePickerOverlay');
        if (overlay) overlay.remove();
    };

    window._videoUploadSceneImage = async function (sceneNum, file) {
        if (!file) return;

        const thumbEl = document.getElementById(`scene-thumb-${sceneNum}`);
        thumbEl.innerHTML = '<span style="color:#a78bfa">Uploading...</span>';
        thumbEl.style.display = 'block';

        try {
            const formData = new FormData();
            formData.append('file', file);

            const resp = await fetch('/api/video/scene-image', {
                method: 'POST',
                body: formData,
            });
            const data = await resp.json();
            if (data.error) throw new Error(data.error);

            const localUrl = URL.createObjectURL(file);
            sceneImages[sceneNum] = { filename: data.filename, url: localUrl };
            try { localStorage.setItem('video_scene_images', JSON.stringify(sceneImages)); } catch (e) { }
            thumbEl.innerHTML = `<img src="${localUrl}" alt="Scene ${sceneNum}">`;
            document.getElementById(`scene-clear-${sceneNum}`).style.display = 'inline-block';
            const card = document.getElementById(`scene-card-${sceneNum}`);
            const btn = card.querySelector('.btn-scene-image');
            btn.textContent = '🔄 Change';
        } catch (e) {
            thumbEl.innerHTML = `<span style="color:#ef4444">${e.message}</span>`;
        }

        // Close picker
        const overlay = document.getElementById('videoImagePickerOverlay');
        if (overlay) overlay.remove();
    };

    window._videoClearImage = function (sceneNum) {
        delete sceneImages[sceneNum];
        try { localStorage.setItem('video_scene_images', JSON.stringify(sceneImages)); } catch (e) { }
        const thumbEl = document.getElementById(`scene-thumb-${sceneNum}`);
        thumbEl.innerHTML = '';
        thumbEl.style.display = 'none';
        document.getElementById(`scene-clear-${sceneNum}`).style.display = 'none';
        const card = document.getElementById(`scene-card-${sceneNum}`);
        const btn = card.querySelector('.btn-scene-image');
        btn.textContent = '📎 Image';
    };


    // ===================================================================
    // Character Management
    // ===================================================================
    let _editingChar = null;

    async function loadCharacters() {
        try {
            const resp = await fetch('/api/video/characters');
            const data = await resp.json();
            const grid = $('videoCharactersGrid');
            if (!grid) return;

            const chars = data.characters || [];
            if (chars.length === 0) {
                grid.innerHTML = '<div class="video-char-empty">No characters yet \u2014 add one to ensure consistent faces across scenes</div>';
                return;
            }

            grid.innerHTML = chars.map(c => {
                const hasImg = c.images && c.images.length > 0;
                const dir = c.dir || c.name;
                const thumbHtml = hasImg
                    ? '<img class="video-char-card-thumb" src="/api/video/characters/' + encodeURIComponent(dir) + '/image/' + encodeURIComponent(c.images[0]) + '" alt="' + c.name + '">'
                    : '<div class="video-char-card-placeholder">\uD83D\uDC64</div>';
                return '<div class="video-char-card" onclick="window._videoEditCharacter(\'' + c.name.replace(/'/g, "\\'") + '\')">' + thumbHtml + '<div class="video-char-card-name">' + c.name + '</div><div class="video-char-card-count">' + (c.image_count || 0) + ' img</div></div>';
            }).join('');
        } catch (e) {
            console.warn('Failed to load characters:', e);
        }
    }

    window._videoAddCharacter = function () {
        _editingChar = null;
        $('videoCharModalTitle').textContent = 'New Character';
        $('videoCharName').value = '';
        $('videoCharDesc').value = '';
        $('videoCharImages').innerHTML = '';
        $('videoCharDeleteBtn').style.display = 'none';
        $('videoCharName').disabled = false;
        $('videoCharModal').style.display = 'flex';
    };

    window._videoEditCharacter = async function (name) {
        _editingChar = name;
        $('videoCharModalTitle').textContent = 'Edit Character';
        $('videoCharDeleteBtn').style.display = 'inline-flex';
        $('videoCharName').disabled = true;
        $('videoCharModal').style.display = 'flex';

        try {
            const resp = await fetch('/api/video/characters');
            const data = await resp.json();
            const char = (data.characters || []).find(c => c.name === name);
            if (!char) return;

            $('videoCharName').value = char.name;
            $('videoCharDesc').value = char.description || '';

            const imgsDiv = $('videoCharImages');
            const dir = char.dir || name;
            imgsDiv.innerHTML = (char.images || []).map(function (img, i) {
                return '<div class="video-char-img-wrapper"><img class="video-char-img-thumb" src="/api/video/characters/' + encodeURIComponent(dir) + '/image/' + encodeURIComponent(img) + '" alt="ref ' + (i + 1) + '"><button class="video-char-img-remove" onclick="window._videoRemoveCharImage(\'' + name.replace(/'/g, "\\'") + '\', ' + i + ')">\u2715</button></div>';
            }).join('');
        } catch (e) {
            console.warn('Failed to load character:', e);
        }
    };

    window._videoCloseCharModal = function (event) {
        if (event && event.target !== event.currentTarget) return;
        $('videoCharModal').style.display = 'none';
    };

    window._videoSaveCharacter = async function () {
        var name = $('videoCharName').value.trim();
        var desc = $('videoCharDesc').value.trim();
        if (!name) return alert('Name is required');

        if (_editingChar) {
            // Update existing character description
            try {
                var resp = await fetch('/api/video/characters/' + encodeURIComponent(_editingChar), {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ description: desc }),
                });
                var data = await resp.json();
                if (data.error) return alert(data.error);
            } catch (e) {
                return alert('Failed to update character: ' + e.message);
            }
        } else {
            // Create new character
            try {
                var resp = await fetch('/api/video/characters', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: name, description: desc }),
                });
                var data = await resp.json();
                if (data.error) return alert(data.error);
            } catch (e) {
                return alert('Failed to create character: ' + e.message);
            }
        }

        $('videoCharModal').style.display = 'none';
        loadCharacters();
    };

    window._videoDeleteCharacter = async function () {
        if (!_editingChar) return;
        if (!confirm('Delete character "' + _editingChar + '"?')) return;

        try {
            await fetch('/api/video/characters/' + encodeURIComponent(_editingChar), { method: 'DELETE' });
        } catch (e) {
            console.warn('Delete failed:', e);
        }

        $('videoCharModal').style.display = 'none';
        loadCharacters();
    };

    window._videoCharUploadImage = async function (input) {
        var file = input.files[0];
        if (!file) return;

        var charName = _editingChar || $('videoCharName').value.trim();
        if (!charName) return alert('Save the character first before uploading images');

        if (!_editingChar) {
            var desc = $('videoCharDesc').value.trim();
            var createResp = await fetch('/api/video/characters', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: charName, description: desc }),
            });
            var createData = await createResp.json();
            if (createData.error) return alert(createData.error);
            _editingChar = charName;
            $('videoCharName').disabled = true;
            $('videoCharDeleteBtn').style.display = 'inline-flex';
        }

        var formData = new FormData();
        formData.append('file', file);

        try {
            var resp2 = await fetch('/api/video/characters/' + encodeURIComponent(charName) + '/image', {
                method: 'POST',
                body: formData,
            });
            var data2 = await resp2.json();
            if (data2.error) return alert(data2.error);
            window._videoEditCharacter(charName);
        } catch (e) {
            alert('Upload failed: ' + e.message);
        }

        input.value = '';
    };

    window._videoRemoveCharImage = async function (name, index) {
        try {
            await fetch('/api/video/characters/' + encodeURIComponent(name) + '/image/' + index, { method: 'DELETE' });
            window._videoEditCharacter(name);
        } catch (e) {
            console.warn('Remove image failed:', e);
        }
    };


    // Auto-init if visible
    if ($('tabVideo') && $('tabVideo').style.display !== 'none') {
        window.initVideoTab();
    }
})();
