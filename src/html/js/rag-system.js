/**
 * RAG System - Browser-based Retrieval-Augmented Generation
 * Manages knowledge base of corrections with semantic embeddings
 */

export class BrowserRAG {
    constructor() {
        this.embedder = null;
        this.knowledgeBase = [];
        this.isInitialized = false;
    }

    async initialize() {
        try {
            console.log('🔄 Initializing RAG embeddings model (this may take a minute)...');

            // Use multilingual-e5-small for browser (supports German, ~130MB)
            this.embedder = await window.transformersPipeline(
                'feature-extraction',
                'Xenova/multilingual-e5-small'
            );

            console.log('✅ Embedding model loaded');
            this.loadFromStorage();
            this.isInitialized = true;

            console.log(`✅ RAG system initialized with ${this.knowledgeBase.length} examples`);

        } catch (error) {
            console.error('❌ RAG initialization failed:', error);
            throw error;
        }
    }

    loadFromStorage() {
        const stored = localStorage.getItem('transcriptionCorrections');
        if (stored) {
            try {
                const corrections = JSON.parse(stored);

                // Load corrections without embeddings (generate on-demand)
                this.knowledgeBase = corrections.map(c => ({
                    id: c.id,
                    asrOutput: c.originalText || c.asrOutput,
                    correctedText: c.correctedText,
                    metadata: {
                        serverName: c.serverName,
                        audioFile: c.audioFile,
                        timestamp: c.timestamp,
                        errorPatterns: c.errorPatterns || []
                    },
                    embedding: null // Generate on-demand during retrieval
                }));

                console.log(`📚 Loaded ${this.knowledgeBase.length} corrections from storage`);

            } catch (error) {
                console.error('Error loading corrections from storage:', error);
                this.knowledgeBase = [];
            }
        }
    }

    saveToStorage() {
        const toSave = this.knowledgeBase.map(item => ({
            id: item.id,
            originalText: item.asrOutput,
            correctedText: item.correctedText,
            serverName: item.metadata.serverName,
            audioFile: item.metadata.audioFile,
            timestamp: item.metadata.timestamp,
            errorPatterns: item.metadata.errorPatterns
            // Don't save embeddings - too large for localStorage
        }));

        localStorage.setItem('transcriptionCorrections', JSON.stringify(toSave));
    }

    async addCorrection(asrOutput, correctedText, metadata = {}) {
        const correction = {
            id: Date.now() + '-' + Math.random().toString(36).substr(2, 9),
            asrOutput: asrOutput,
            correctedText: correctedText,
            metadata: {
                serverName: metadata.serverName || 'unknown',
                audioFile: metadata.audioFile || 'unknown',
                timestamp: new Date().toISOString(),
                errorPatterns: metadata.errorPatterns || []
            },
            embedding: null
        };

        // Generate embedding if RAG is initialized
        if (this.isInitialized && this.embedder) {
            try {
                const embedding = await this.embedder(asrOutput, {
                    pooling: 'mean',
                    normalize: true
                });
                correction.embedding = Array.from(embedding.data);
                console.log(`✅ Added correction with embedding (total: ${this.knowledgeBase.length + 1})`);
            } catch (error) {
                console.warn('Failed to generate embedding:', error);
            }
        } else {
            console.log(`✅ Added correction without embedding (total: ${this.knowledgeBase.length + 1})`);
        }

        this.knowledgeBase.push(correction);
        this.saveToStorage();
    }

    async retrieveSimilar(query, topK = 3) {
        if (!this.isInitialized) {
            console.warn('⚠️ RAG system not initialized');
            return [];
        }

        if (this.knowledgeBase.length === 0) {
            console.warn('⚠️ Knowledge base is empty');
            return [];
        }

        try {
            // Generate query embedding
            const queryEmbedding = await this.embedder(query, {
                pooling: 'mean',
                normalize: true
            });

            const queryVector = Array.from(queryEmbedding.data);

            // Generate embeddings for items that don't have them
            for (let item of this.knowledgeBase) {
                if (!item.embedding) {
                    const embedding = await this.embedder(item.asrOutput, {
                        pooling: 'mean',
                        normalize: true
                    });
                    item.embedding = Array.from(embedding.data);
                }
            }

            // Cosine similarity search
            const similarities = this.knowledgeBase.map((item) => ({
                ...item,
                similarity: this.cosineSimilarity(queryVector, item.embedding)
            }));

            // Sort by similarity and return top-k
            const topResults = similarities
                .sort((a, b) => b.similarity - a.similarity)
                .slice(0, topK);

            console.log(`🔍 Retrieved ${topResults.length} similar examples (top similarity: ${(topResults[0]?.similarity || 0).toFixed(3)})`);

            return topResults;

        } catch (error) {
            console.error('RAG retrieval error:', error);
            return [];
        }
    }

    cosineSimilarity(a, b) {
        const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
        const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
        const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
        return dotProduct / (magA * magB);
    }

    getStats() {
        return {
            totalExamples: this.knowledgeBase.length,
            isInitialized: this.isInitialized,
            hasEmbeddings: this.knowledgeBase.some(item => item.embedding !== null)
        };
    }

    clear() {
        this.knowledgeBase = [];
        localStorage.removeItem('transcriptionCorrections');
    }
}

/**
 * German Text Processor - Detects common error patterns
 */
export class GermanTextProcessor {
    detectErrorPatterns(asrOutput, correctedText) {
        const patterns = [];

        // Simple word-level comparison
        const asrWords = asrOutput.split(' ');
        const correctedWords = correctedText.split(' ');

        asrWords.forEach((word, i) => {
            const correctedWord = correctedWords[i];
            if (correctedWord &&
                word.toLowerCase() === correctedWord.toLowerCase() &&
                word !== correctedWord) {
                patterns.push({
                    type: 'capitalization',
                    from: word,
                    to: correctedWord
                });
            }
        });

        return patterns;
    }
}
