/**
 * LLM Correctors - Local and Cloud-based text correction
 */

/**
 * LocalLLMCorrector - Browser-based correction using Transformers.js
 * Uses Qwen2.5-0.5B-Instruct model for German text correction
 */
export class LocalLLMCorrector {
    constructor() {
        this.generator = null;
        this.isInitialized = false;
        this.modelType = 'flan-t5';
        this.modelName = 'Flan-T5-Small';
    }

    async initialize() {
        console.log('🔄 Initializing Local LLM (Flan-T5-Small, ~300MB download)...');
        console.log('⏳ This may take a minute on first load...');

        try {
            // Use Flan-T5-Small - works reliably with Transformers.js
            // Good for text correction tasks, supports German
            this.generator = await window.transformersPipeline(
                'text2text-generation',
                'Xenova/flan-t5-small'
            );
            this.modelType = 'flan-t5';
            this.modelName = 'Flan-T5-Small';
            console.log('✅ Loaded Flan-T5-Small');

            this.isInitialized = true;
            console.log('✅ Local LLM ready');
            return true;

        } catch (error) {
            console.error('❌ Local LLM initialization failed:', error);
            this.isInitialized = false;
            return false;
        }
    }

    async correct(asrText, similarExamples, germanPrompt) {
        if (!this.isInitialized) {
            throw new Error('Local LLM not initialized. Call initialize() first.');
        }

        try {
            // Flan-T5 works best with simple, direct task descriptions
            const simplePrompt = `Correct this German transcription, fixing capitalization, spelling, and compound words:\n\n${asrText}`;

            const result = await this.generator(simplePrompt, {
                max_new_tokens: 200,
                temperature: 0.3
            });

            const correctedText = result[0].generated_text.trim();

            return {
                correctedText: correctedText,
                confidence: 0.7, // Local models have lower confidence
                provider: 'local',
                model: this.modelName
            };

        } catch (error) {
            console.error('❌ Local LLM correction failed:', error);
            throw error;
        }
    }
}

/**
 * APILLMCorrector - Cloud-based correction using OpenAI, Anthropic, or Mistral
 */
export class APILLMCorrector {
    constructor(provider, apiKey, model) {
        this.provider = provider;
        this.apiKey = apiKey;
        this.model = model;
    }

    async correct(asrText, similarExamples, germanPrompt) {
        if (!this.apiKey) {
            throw new Error('API key not configured');
        }

        try {
            let response;

            switch (this.provider) {
                case 'openai':
                    response = await this.callOpenAI(germanPrompt);
                    break;
                case 'anthropic':
                    response = await this.callAnthropic(germanPrompt);
                    break;
                case 'mistral':
                    response = await this.callMistral(germanPrompt);
                    break;
                default:
                    throw new Error(`Unknown provider: ${this.provider}`);
            }

            return response;

        } catch (error) {
            console.error(`❌ ${this.provider} API call failed:`, error);
            throw error;
        }
    }

    async callOpenAI(prompt) {
        const response = await fetch('https://api.openai.com/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.apiKey}`
            },
            body: JSON.stringify({
                model: this.model,
                messages: [
                    {
                        role: 'system',
                        content: 'You are a German transcription correction expert. Fix ASR errors while preserving the original meaning.'
                    },
                    {
                        role: 'user',
                        content: prompt
                    }
                ],
                temperature: 0.3,
                max_tokens: 500
            })
        });

        if (!response.ok) {
            throw new Error(`OpenAI API error: ${response.statusText}`);
        }

        const data = await response.json();
        const correctedText = data.choices[0].message.content.trim();

        return {
            correctedText: correctedText,
            confidence: 0.9,
            provider: 'openai',
            model: this.model
        };
    }

    async callAnthropic(prompt) {
        const response = await fetch('https://api.anthropic.com/v1/messages', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-api-key': this.apiKey,
                'anthropic-version': '2023-06-01'
            },
            body: JSON.stringify({
                model: this.model,
                max_tokens: 500,
                temperature: 0.3,
                messages: [
                    {
                        role: 'user',
                        content: prompt
                    }
                ]
            })
        });

        if (!response.ok) {
            throw new Error(`Anthropic API error: ${response.statusText}`);
        }

        const data = await response.json();
        const correctedText = data.content[0].text.trim();

        return {
            correctedText: correctedText,
            confidence: 0.95,
            provider: 'anthropic',
            model: this.model
        };
    }

    async callMistral(prompt) {
        const response = await fetch('https://api.mistral.ai/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.apiKey}`
            },
            body: JSON.stringify({
                model: this.model,
                messages: [
                    {
                        role: 'system',
                        content: 'You are a German transcription correction expert. Fix ASR errors while preserving the original meaning.'
                    },
                    {
                        role: 'user',
                        content: prompt
                    }
                ],
                temperature: 0.3,
                max_tokens: 500
            })
        });

        if (!response.ok) {
            throw new Error(`Mistral API error: ${response.statusText}`);
        }

        const data = await response.json();
        const correctedText = data.choices[0].message.content.trim();

        return {
            correctedText: correctedText,
            confidence: 0.9,
            provider: 'mistral',
            model: this.model
        };
    }
}

/**
 * Build a German-specific correction prompt with RAG examples
 */
export function buildGermanCorrectionPrompt(asrText, similarExamples) {
    let prompt = `Du bist ein Experte für die Korrektur deutscher Transkriptionen. Korrigiere die folgenden ASR-Fehler:

1. Rechtschreibung und Grammatik
2. Deutsche Großschreibung (Substantive, etc.)
3. Zusammengesetzte Wörter
4. Zahlen und Datumsangaben
5. Umlaute (ä, ö, ü, ß)

`;

    // Add similar examples if available
    if (similarExamples && similarExamples.length > 0) {
        prompt += `BEISPIELE von vorherigen Korrekturen:\n\n`;

        similarExamples.forEach((example, idx) => {
            prompt += `Beispiel ${idx + 1}:\n`;
            prompt += `ASR: "${example.asrOutput}"\n`;
            prompt += `Korrigiert: "${example.correctedText}"\n\n`;
        });
    }

    prompt += `Jetzt korrigiere diesen Text:\n\nASR: "${asrText}"\n\nKorrigiert:`;

    return prompt;
}
