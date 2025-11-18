/**
 * RAG UI Functions - User interface for corrections and knowledge base management
 */

import { buildGermanCorrectionPrompt } from './llm-correctors.js';

// Default RAG settings
export const defaultRAGSettings = {
    provider: 'local',
    apiKey: '',
    apiModel: '',
    topK: 3,
    minSimilarity: 0.5,
    autoCorrect: false,
    confidenceThreshold: 0.8
};

// Model options for each provider
export const providerModels = {
    openai: ['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo'],
    anthropic: ['claude-3-5-haiku-20241022', 'claude-3-5-sonnet-20241022'],
    mistral: ['mistral-small-latest', 'mistral-medium-latest', 'mistral-large-latest']
};

/**
 * Manual Edit Functions
 */
export function manualEdit(serverName) {
    const textEl = document.getElementById(`text-${serverName}`);
    const originalText = textEl.textContent.trim();

    // Store original text
    textEl.dataset.originalText = originalText;

    // Make text editable
    textEl.contentEditable = true;
    textEl.focus();

    // Change button to Save
    const btn = document.getElementById(`edit-${serverName}`);
    btn.textContent = '💾 Save';
    btn.className = 'edit-btn success';
    btn.onclick = () => window.saveManualEdit(serverName);
}

export function saveManualEdit(serverName) {
    const textEl = document.getElementById(`text-${serverName}`);
    const editedText = textEl.textContent.trim();
    const originalText = textEl.dataset.originalText;

    textEl.contentEditable = false;
    textEl.className = 'transcription-text edited-segment';

    // Reset button
    const btn = document.getElementById(`edit-${serverName}`);
    btn.textContent = '✏️ Edit';
    btn.className = 'edit-btn secondary';
    btn.onclick = () => window.manualEdit(serverName);

    // Save to localStorage if changed
    if (editedText !== originalText) {
        window.saveCorrection(serverName, originalText, editedText);
        console.log(`✅ Manual edit saved for ${serverName}`);
    }

    delete textEl.dataset.originalText;
}

/**
 * AI Correction Workflow
 */
export async function aiCorrect(serverName, ragSystem, textProcessor, llmCorrector, LocalLLMCorrector, APILLMCorrector) {
    const textEl = document.getElementById(`text-${serverName}`);
    const originalText = textEl.textContent.trim();
    const btn = document.getElementById(`ai-correct-${serverName}`);

    if (!originalText || originalText === 'Waiting for audio...') {
        alert('⚠️ No transcription available to correct');
        return;
    }

    try {
        // Show loading state
        btn.disabled = true;
        btn.textContent = '⏳ Correcting...';

        // Load settings
        const settings = JSON.parse(localStorage.getItem('ragSettings') || JSON.stringify(defaultRAGSettings));

        // Initialize LLM corrector if needed
        if (!llmCorrector) {
            if (settings.provider === 'local') {
                llmCorrector = new LocalLLMCorrector();
                const initialized = await llmCorrector.initialize();
                if (!initialized) {
                    throw new Error('Failed to initialize local LLM');
                }
            } else {
                llmCorrector = new APILLMCorrector(
                    settings.provider,
                    settings.apiKey,
                    settings.apiModel
                );
            }
            // Store globally
            window.llmCorrector = llmCorrector;
        }

        // Step 1: Retrieve similar examples from RAG system
        console.log('🔍 Step 1: Retrieving similar examples from knowledge base...');
        let similarExamples = [];

        if (ragSystem && ragSystem.isInitialized && ragSystem.knowledgeBase.length > 0) {
            similarExamples = await ragSystem.retrieveSimilar(originalText, settings.topK);

            // Filter by minimum similarity
            similarExamples = similarExamples.filter(ex => ex.similarity >= settings.minSimilarity);

            console.log(`📚 Found ${similarExamples.length} similar examples (threshold: ${settings.minSimilarity})`);

            if (similarExamples.length > 0) {
                similarExamples.forEach((ex, idx) => {
                    console.log(`  ${idx + 1}. Similarity: ${ex.similarity.toFixed(3)} - "${ex.asrOutput.substring(0, 50)}..."`);
                });
            }
        } else {
            console.log('📚 No knowledge base examples available (starting fresh)');
        }

        // Step 2: Build German prompt with RAG examples
        console.log('📝 Step 2: Building German correction prompt...');
        const prompt = buildGermanCorrectionPrompt(originalText, similarExamples);

        // Step 3: Call LLM for correction
        console.log(`🤖 Step 3: Calling ${settings.provider} LLM for correction...`);
        const result = await window.llmCorrector.correct(originalText, similarExamples, prompt);

        console.log(`✅ LLM correction received (confidence: ${result.confidence})`);
        console.log(`   Original: "${originalText}"`);
        console.log(`   Corrected: "${result.correctedText}"`);

        // Step 4: Show diff and let user accept/reject
        showCorrectionDiff(serverName, originalText, result.correctedText, result);

    } catch (error) {
        console.error('❌ AI correction failed:', error);
        alert(`❌ AI correction failed: ${error.message}`);

    } finally {
        // Reset button state
        btn.disabled = false;
        btn.textContent = '🤖 AI Correct';
    }
}

function showCorrectionDiff(serverName, originalText, correctedText, result) {
    const diffEl = document.getElementById(`diff-${serverName}`);

    // Store data for accept/reject
    diffEl.dataset.originalText = originalText;
    diffEl.dataset.correctedText = correctedText;

    // Build confidence badge
    const confidenceBadge = result.confidence >= 0.9 ?
        `<span class="confidence-badge high">High Confidence: ${(result.confidence * 100).toFixed(0)}%</span>` :
        result.confidence >= 0.7 ?
        `<span class="confidence-badge medium">Medium Confidence: ${(result.confidence * 100).toFixed(0)}%</span>` :
        `<span class="confidence-badge low">Low Confidence: ${(result.confidence * 100).toFixed(0)}%</span>`;

    diffEl.innerHTML = `
        <div style="margin-bottom: 15px;">
            <strong>🤖 AI Correction Result</strong>
            ${confidenceBadge}
            <span style="margin-left: 10px; color: #666; font-size: 0.9em;">Provider: ${result.provider} (${result.model})</span>
        </div>

        <div class="diff-original">
            <strong>Original ASR:</strong>
            <div>${escapeHtml(originalText)}</div>
        </div>

        <div class="diff-corrected">
            <strong>AI Corrected:</strong>
            <div>${escapeHtml(correctedText)}</div>
        </div>

        <div style="margin-top: 15px; display: flex; gap: 10px;">
            <button class="edit-btn success" onclick="window.acceptCorrection('${serverName}')">✅ Accept</button>
            <button class="edit-btn danger" onclick="window.rejectCorrection('${serverName}')">❌ Reject</button>
            <button class="edit-btn secondary" onclick="window.editBeforeAccept('${serverName}')">✏️ Edit Before Accept</button>
        </div>
    `;

    diffEl.style.display = 'block';

    // Scroll to diff
    diffEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

export async function acceptCorrection(serverName) {
    const diffEl = document.getElementById(`diff-${serverName}`);
    const textEl = document.getElementById(`text-${serverName}`);
    const originalText = diffEl.dataset.originalText;
    const correctedText = diffEl.dataset.correctedText;

    // Update text
    textEl.textContent = correctedText;
    textEl.className = 'transcription-text edited-segment';

    // Save to knowledge base
    await window.saveCorrection(serverName, originalText, correctedText);

    // Hide diff
    diffEl.style.display = 'none';

    console.log(`✅ Accepted AI correction for ${serverName}`);
}

export function rejectCorrection(serverName) {
    const diffEl = document.getElementById(`diff-${serverName}`);
    diffEl.style.display = 'none';

    console.log(`❌ Rejected AI correction for ${serverName}`);
}

export function editBeforeAccept(serverName) {
    const diffEl = document.getElementById(`diff-${serverName}`);
    const textEl = document.getElementById(`text-${serverName}`);
    const originalText = diffEl.dataset.originalText;
    const correctedText = diffEl.dataset.correctedText;

    // Set text to AI suggestion
    textEl.textContent = correctedText;

    // Hide diff
    diffEl.style.display = 'none';

    // Store original for later
    textEl.dataset.originalText = originalText;

    // Make editable
    textEl.contentEditable = true;
    textEl.focus();

    // Change button to Save
    const btn = document.getElementById(`edit-${serverName}`);
    btn.textContent = '💾 Save';
    btn.className = 'edit-btn success';
    btn.onclick = () => window.saveManualEdit(serverName);

    console.log(`✏️ Editing AI suggestion for ${serverName}`);
}

/**
 * Knowledge Base Management UI
 */
export function toggleKBPanel() {
    const panel = document.getElementById('kbPanelContent');
    const icon = document.getElementById('kb-toggle-icon');

    if (panel.style.display === 'none') {
        panel.style.display = 'block';
        icon.textContent = '▲';
        window.refreshKBStats();
    } else {
        panel.style.display = 'none';
        icon.textContent = '▼';
    }
}

export function refreshKBStats() {
    const ragSystem = window.ragSystem;
    if (!ragSystem) {
        console.warn('RAG system not initialized');
        return;
    }

    const stats = ragSystem.getStats();

    document.getElementById('kb-count').textContent = stats.totalExamples;
    document.getElementById('kb-embeddings-status').textContent = stats.isInitialized ? '✅ Loaded' : '⏳ Loading...';
    document.getElementById('kb-updated').textContent =
        ragSystem.knowledgeBase.length > 0
            ? new Date(Math.max(...ragSystem.knowledgeBase.map(item => new Date(item.metadata.timestamp)))).toLocaleString()
            : 'Never';

    // Update list
    renderKBList();
}

export function renderKBList() {
    const ragSystem = window.ragSystem;
    const listEl = document.getElementById('kbList');
    if (!ragSystem || ragSystem.knowledgeBase.length === 0) {
        listEl.innerHTML = '<p style="color: #666; font-style: italic; text-align: center; padding: 20px;">No corrections in knowledge base yet. Edit a transcription to add your first example!</p>';
        return;
    }

    // Show most recent 10
    const recent = ragSystem.knowledgeBase.slice(-10).reverse();

    listEl.innerHTML = recent.map((item, idx) => `
        <div class="kb-item">
            <div style="font-size: 0.85em; color: #666; margin-bottom: 8px;">
                <strong>#${ragSystem.knowledgeBase.length - idx}</strong> •
                ${new Date(item.metadata.timestamp).toLocaleString()} •
                ${item.metadata.serverName || 'unknown'}
                ${item.embedding ? ' • <span style="color: #28a745;">✓ Has embedding</span>' : ' • <span style="color: #dc3545;">⚠ No embedding</span>'}
            </div>
            <div style="margin: 8px 0; padding: 8px; background: #f8d7da; border-radius: 4px;">
                <strong style="font-size: 0.85em; color: #721c24;">Original ASR:</strong><br>
                <span style="color: #333;">${escapeHtml(item.asrOutput)}</span>
            </div>
            <div style="margin: 8px 0; padding: 8px; background: #d4edda; border-radius: 4px;">
                <strong style="font-size: 0.85em; color: #155724;">Corrected:</strong><br>
                <span style="color: #333;">${escapeHtml(item.correctedText)}</span>
            </div>
        </div>
    `).join('');
}

export function exportKnowledgeBase() {
    const ragSystem = window.ragSystem;
    if (!ragSystem) {
        alert('RAG system not initialized');
        return;
    }

    const exportData = {
        version: '1.0',
        exportDate: new Date().toISOString(),
        totalExamples: ragSystem.knowledgeBase.length,
        corrections: ragSystem.knowledgeBase.map(item => ({
            id: item.id,
            asrOutput: item.asrOutput,
            correctedText: item.correctedText,
            metadata: item.metadata
        }))
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `rag_knowledge_base_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);

    console.log(`📤 Exported ${exportData.totalExamples} examples`);
}

export function importKnowledgeBase() {
    const ragSystem = window.ragSystem;
    if (!ragSystem) {
        alert('RAG system not initialized');
        return;
    }

    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';

    input.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        try {
            const text = await file.text();
            const data = JSON.parse(text);

            if (!data.corrections || !Array.isArray(data.corrections)) {
                throw new Error('Invalid knowledge base format');
            }

            // Merge with existing KB
            let imported = 0;
            for (const correction of data.corrections) {
                await ragSystem.addCorrection(
                    correction.asrOutput,
                    correction.correctedText,
                    correction.metadata || {}
                );
                imported++;
            }

            alert(`✅ Imported ${imported} examples into knowledge base`);
            window.refreshKBStats();

        } catch (error) {
            console.error('Import error:', error);
            alert('❌ Import failed: ' + error.message);
        }
    };

    input.click();
}

export function clearKnowledgeBase() {
    const ragSystem = window.ragSystem;
    if (!ragSystem) {
        alert('RAG system not initialized');
        return;
    }

    if (!confirm('⚠️ This will delete all correction examples from the knowledge base. Are you sure?')) {
        return;
    }

    ragSystem.clear();
    window.refreshKBStats();

    alert('✅ Knowledge base cleared');
}

/**
 * Settings Modal Functions
 */
export function openRAGSettings() {
    // Load saved settings
    const settings = JSON.parse(localStorage.getItem('ragSettings') || JSON.stringify(defaultRAGSettings));

    // Populate form
    document.querySelector(`input[name="llmProvider"][value="${settings.provider}"]`).checked = true;
    document.getElementById('apiKey').value = settings.apiKey || '';
    document.getElementById('topK').value = settings.topK;
    document.getElementById('minSimilarity').value = settings.minSimilarity;
    document.getElementById('autoCorrect').checked = settings.autoCorrect;
    document.getElementById('confidenceThreshold').value = settings.confidenceThreshold;

    // Update provider-specific settings
    updateProviderSettings();

    // Show modal
    document.getElementById('ragSettingsModal').style.display = 'flex';
}

export function closeRAGSettings() {
    document.getElementById('ragSettingsModal').style.display = 'none';
}

export function updateProviderSettings() {
    const provider = document.querySelector('input[name="llmProvider"]:checked').value;
    const apiSettings = document.getElementById('apiSettings');
    const modelSelect = document.getElementById('apiModel');

    if (provider === 'local') {
        apiSettings.style.display = 'none';
    } else {
        apiSettings.style.display = 'block';

        // Populate model dropdown
        const models = providerModels[provider] || [];
        modelSelect.innerHTML = models.map(model =>
            `<option value="${model}">${model}</option>`
        ).join('');

        // Load saved model if available
        const settings = JSON.parse(localStorage.getItem('ragSettings') || '{}');
        if (settings.apiModel && models.includes(settings.apiModel)) {
            modelSelect.value = settings.apiModel;
        }
    }
}

export function updateAutoCorrectSettings() {
    const autoCorrect = document.getElementById('autoCorrect').checked;
    document.getElementById('autoCorrectSettings').style.display = autoCorrect ? 'block' : 'none';
}

export function saveRAGSettings() {
    const provider = document.querySelector('input[name="llmProvider"]:checked').value;
    const settings = {
        provider: provider,
        apiKey: provider === 'local' ? '' : document.getElementById('apiKey').value,
        apiModel: provider === 'local' ? '' : document.getElementById('apiModel').value,
        topK: parseInt(document.getElementById('topK').value),
        minSimilarity: parseFloat(document.getElementById('minSimilarity').value),
        autoCorrect: document.getElementById('autoCorrect').checked,
        confidenceThreshold: parseFloat(document.getElementById('confidenceThreshold').value)
    };

    // Validate API key for cloud providers
    if (provider !== 'local' && !settings.apiKey) {
        alert('⚠️ Please enter an API key for the selected provider');
        return;
    }

    // Save to localStorage
    localStorage.setItem('ragSettings', JSON.stringify(settings));

    console.log('✅ RAG settings saved:', settings);
    alert('✅ Settings saved successfully!');

    closeRAGSettings();
}

/**
 * Utility Functions
 */
export function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
