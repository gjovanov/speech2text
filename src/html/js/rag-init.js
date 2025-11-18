/**
 * RAG Initialization - Sets up the RAG system and makes functions globally available
 */

import { BrowserRAG, GermanTextProcessor } from './rag-system.js';
import { LocalLLMCorrector, APILLMCorrector } from './llm-correctors.js';
import {
    manualEdit,
    saveManualEdit,
    aiCorrect,
    acceptCorrection,
    rejectCorrection,
    editBeforeAccept,
    toggleKBPanel,
    refreshKBStats,
    renderKBList,
    exportKnowledgeBase,
    importKnowledgeBase,
    clearKnowledgeBase,
    openRAGSettings,
    closeRAGSettings,
    updateProviderSettings,
    updateAutoCorrectSettings,
    saveRAGSettings,
    escapeHtml
} from './rag-ui.js';

// Global instances
window.ragSystem = null;
window.textProcessor = null;
window.llmCorrector = null;

/**
 * Initialize RAG system on page load
 */
export async function initializeRAGSystem() {
    try {
        console.log('🔄 Initializing RAG system...');

        window.ragSystem = new BrowserRAG();
        window.textProcessor = new GermanTextProcessor();

        // Initialize embeddings in background (don't block page load)
        setTimeout(() => {
            window.ragSystem.initialize().catch(err => {
                console.warn('RAG embeddings not loaded:', err);
            });
        }, 2000);

        console.log('✅ RAG system classes created');

    } catch (error) {
        console.error('❌ RAG system creation failed:', error);
    }
}

/**
 * Save correction to localStorage and RAG system
 */
export async function saveCorrection(serverName, originalText, correctedText) {
    try {
        // Detect error patterns
        const patterns = window.textProcessor ? window.textProcessor.detectErrorPatterns(originalText, correctedText) : [];

        // Add to RAG system if available
        if (window.ragSystem) {
            await window.ragSystem.addCorrection(originalText, correctedText, {
                serverName: serverName,
                audioFile: document.getElementById('audioSelect')?.value || 'unknown',
                errorPatterns: patterns
            });
        } else {
            // Fallback to manual localStorage save
            let corrections = JSON.parse(localStorage.getItem('transcriptionCorrections') || '[]');
            corrections.push({
                id: Date.now() + '-' + Math.random().toString(36).substr(2, 9),
                serverName: serverName,
                originalText: originalText,
                correctedText: correctedText,
                timestamp: new Date().toISOString(),
                audioFile: document.getElementById('audioSelect')?.value || 'unknown'
            });
            localStorage.setItem('transcriptionCorrections', JSON.stringify(corrections));
        }

        console.log(`💾 Saved correction to RAG knowledge base`);
    } catch (error) {
        console.error('Error saving correction:', error);
    }
}

/**
 * Wrapper for aiCorrect to pass global instances
 */
export async function aiCorrectWrapper(serverName) {
    await aiCorrect(
        serverName,
        window.ragSystem,
        window.textProcessor,
        window.llmCorrector,
        LocalLLMCorrector,
        APILLMCorrector
    );
}

/**
 * Make all functions globally available for onclick handlers
 */
export function setupGlobalFunctions() {
    // Manual editing
    window.manualEdit = manualEdit;
    window.saveManualEdit = saveManualEdit;

    // AI correction
    window.aiCorrect = aiCorrectWrapper;
    window.acceptCorrection = acceptCorrection;
    window.rejectCorrection = rejectCorrection;
    window.editBeforeAccept = editBeforeAccept;

    // Knowledge base management
    window.toggleKBPanel = toggleKBPanel;
    window.refreshKBStats = refreshKBStats;
    window.exportKnowledgeBase = exportKnowledgeBase;
    window.importKnowledgeBase = importKnowledgeBase;
    window.clearKnowledgeBase = clearKnowledgeBase;

    // Settings modal
    window.openRAGSettings = openRAGSettings;
    window.closeRAGSettings = closeRAGSettings;
    window.updateProviderSettings = updateProviderSettings;
    window.updateAutoCorrectSettings = updateAutoCorrectSettings;
    window.saveRAGSettings = saveRAGSettings;

    // Utilities
    window.saveCorrection = saveCorrection;
    window.escapeHtml = escapeHtml;
}

/**
 * Call this on window load
 */
export function initRAG() {
    setupGlobalFunctions();
    initializeRAGSystem();
}
