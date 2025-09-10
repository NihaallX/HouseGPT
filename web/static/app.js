// HouseGPT Web Interface JavaScript

class HouseGPTChat {
    constructor() {
        this.baseUrl = '/api';
        this.isConnected = false;
        this.isVoiceEnabled = false;
        this.sessionStats = {
            messages: 0,
            responses: 0,
            avgResponseTime: 0,
            totalTime: 0
        };
        this.messageHistory = [];
        this.currentRequestId = null;
        this.retryCount = 0;
        this.maxRetries = 3;
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.checkConnection();
        this.loadSessionStats();
        this.autoResizeTextarea();
        this.showWelcomeMessage();
    }

    bindEvents() {
        // Get DOM elements
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.chatMessages = document.getElementById('chatMessages');
        this.voiceToggle = document.getElementById('voiceToggle');
        this.clearButton = document.getElementById('clearChat');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.statusText = document.getElementById('statusText');
        this.statsToggle = document.getElementById('statsToggle');
        this.statsContent = document.getElementById('statsContent');
        this.characterCount = document.getElementById('characterCount');

        // Event listeners
        this.sendButton?.addEventListener('click', () => this.sendMessage());
        this.messageInput?.addEventListener('keydown', (e) => this.handleKeyDown(e));
        this.messageInput?.addEventListener('input', () => this.updateCharacterCount());
        this.voiceToggle?.addEventListener('click', () => this.toggleVoice());
        this.clearButton?.addEventListener('click', () => this.clearChat());
        this.statsToggle?.addEventListener('click', () => this.toggleStats());

        // Quick action buttons
        document.querySelectorAll('.quick-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.messageInput.value = btn.dataset.text;
                this.updateCharacterCount();
                this.messageInput.focus();
            });
        });

        // Auto-resize textarea
        this.messageInput?.addEventListener('input', () => this.autoResizeTextarea());
    }

    handleKeyDown(e) {
        if (e.key === 'Enter') {
            if (e.shiftKey) {
                // Shift+Enter: New line
                return;
            } else {
                // Enter: Send message
                e.preventDefault();
                this.sendMessage();
            }
        }
    }

    autoResizeTextarea() {
        if (!this.messageInput) return;
        
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
    }

    updateCharacterCount() {
        if (!this.characterCount || !this.messageInput) return;
        
        const count = this.messageInput.value.length;
        const maxCount = 1000; // Reasonable limit
        
        this.characterCount.textContent = `${count}/${maxCount}`;
        
        if (count > maxCount * 0.9) {
            this.characterCount.style.color = 'var(--warning-color)';
        } else if (count > maxCount) {
            this.characterCount.style.color = 'var(--error-color)';
        } else {
            this.characterCount.style.color = 'var(--text-muted)';
        }
    }

    async checkConnection() {
        try {
            const response = await fetch(`${this.baseUrl}/health`);
            const data = await response.json();
            
            this.isConnected = data.status === 'healthy';
            this.updateConnectionStatus();
            
            if (this.isConnected) {
                this.retryCount = 0;
            }
        } catch (error) {
            console.error('Connection check failed:', error);
            this.isConnected = false;
            this.updateConnectionStatus();
            
            // Retry connection after delay
            if (this.retryCount < this.maxRetries) {
                this.retryCount++;
                setTimeout(() => this.checkConnection(), 2000 * this.retryCount);
            }
        }
    }

    updateConnectionStatus() {
        if (!this.statusIndicator || !this.statusText) return;
        
        if (this.isConnected) {
            this.statusIndicator.classList.add('connected');
            this.statusText.textContent = 'Connected';
            this.sendButton?.removeAttribute('disabled');
        } else {
            this.statusIndicator.classList.remove('connected');
            this.statusText.textContent = 'Disconnected';
            this.sendButton?.setAttribute('disabled', 'disabled');
        }
    }

    showWelcomeMessage() {
        const welcomeText = `
        👋 **Welcome to HouseGPT!**
        
        I'm Dr. Gregory House, MD. I'm here to provide medical insights with my signature wit and wisdom.
        
        Ask me about:
        • Medical symptoms and conditions
        • Diagnostic puzzles
        • Treatment options
        • Medical philosophy
        
        Remember: I'm an AI assistant for educational purposes. Always consult real medical professionals for actual health concerns.
        
        What medical mystery can I help you solve today?
        `.trim();

        this.addMessage('house', 'Dr. House', welcomeText, new Date(), true);
    }

    async sendMessage() {
        const message = this.messageInput?.value.trim();
        if (!message || !this.isConnected) return;

        // Clear input
        this.messageInput.value = '';
        this.updateCharacterCount();
        this.autoResizeTextarea();

        // Add user message to chat
        const timestamp = new Date();
        this.addMessage('user', 'You', message, timestamp);
        
        // Show typing indicator
        this.showTypingIndicator();
        
        // Update session stats
        this.sessionStats.messages++;
        this.updateSessionStats();

        // Send message to API
        const startTime = Date.now();
        
        try {
            const requestId = Date.now().toString();
            this.currentRequestId = requestId;
            
            const response = await fetch(`${this.baseUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    session_id: this.getSessionId(),
                    voice_enabled: this.isVoiceEnabled
                })
            });

            // Check if this is still the current request
            if (this.currentRequestId !== requestId) {
                return; // Ignore outdated responses
            }

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            
            // Calculate response time
            const responseTime = Date.now() - startTime;
            this.updateResponseTime(responseTime);
            
            // Hide typing indicator
            this.hideTypingIndicator();
            
            // Add response to chat
            this.addMessage('house', 'Dr. House', data.response, new Date(), false, {
                responseTime: responseTime,
                confidence: data.confidence,
                sources: data.sources
            });
            
            // Handle voice if enabled
            if (this.isVoiceEnabled && data.audio_url) {
                this.playAudio(data.audio_url);
            }
            
            // Update session stats
            this.sessionStats.responses++;
            this.updateSessionStats();
            
        } catch (error) {
            console.error('Failed to send message:', error);
            this.hideTypingIndicator();
            this.showError(`Failed to send message: ${error.message}`);
        }
    }

    addMessage(type, sender, text, timestamp, isWelcome = false, metadata = null) {
        if (!this.chatMessages) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message ${isWelcome ? 'welcome-message' : ''}`;
        
        const avatar = type === 'house' ? '🏥' : '👤';
        const formattedTime = timestamp.toLocaleTimeString();
        
        // Convert markdown-style formatting to HTML
        const formattedText = this.formatMessage(text);
        
        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-header">
                    <span class="sender-name">${sender}</span>
                    <span class="message-time">${formattedTime}</span>
                </div>
                <div class="message-text">${formattedText}</div>
                ${metadata ? this.formatMetadata(metadata) : ''}
            </div>
        `;

        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        
        // Store in history
        this.messageHistory.push({
            type,
            sender,
            text,
            timestamp,
            metadata
        });
    }

    formatMessage(text) {
        // Simple markdown-like formatting
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>')
            .replace(/^• (.*$)/gim, '<li>$1</li>')
            .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
    }

    formatMetadata(metadata) {
        const items = [];
        
        if (metadata.responseTime) {
            items.push(`<span class="meta-item">⏱️ ${metadata.responseTime}ms</span>`);
        }
        
        if (metadata.confidence) {
            items.push(`<span class="meta-item">🎯 ${Math.round(metadata.confidence * 100)}%</span>`);
        }
        
        if (metadata.sources && metadata.sources.length > 0) {
            items.push(`<span class="meta-item">📚 ${metadata.sources.length} sources</span>`);
        }
        
        return items.length > 0 ? `<div class="message-meta">${items.join('')}</div>` : '';
    }

    showTypingIndicator() {
        if (!this.chatMessages) return;

        // Remove existing typing indicator
        this.hideTypingIndicator();

        const typingDiv = document.createElement('div');
        typingDiv.className = 'message house-message typing-indicator';
        typingDiv.id = 'typingIndicator';
        
        typingDiv.innerHTML = `
            <div class="message-avatar">🏥</div>
            <div class="typing-content">
                <span>Dr. House is typing</span>
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;

        this.chatMessages.appendChild(typingDiv);
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    scrollToBottom() {
        if (!this.chatMessages) return;
        
        const container = this.chatMessages.parentElement;
        if (container) {
            container.scrollTop = container.scrollHeight;
        }
    }

    toggleVoice() {
        this.isVoiceEnabled = !this.isVoiceEnabled;
        
        if (this.voiceToggle) {
            this.voiceToggle.setAttribute('data-enabled', this.isVoiceEnabled);
            const voiceText = this.voiceToggle.querySelector('.voice-text');
            if (voiceText) {
                voiceText.textContent = this.isVoiceEnabled ? 'Voice On' : 'Voice Off';
            }
        }
        
        this.showToast(
            this.isVoiceEnabled ? 'Voice synthesis enabled' : 'Voice synthesis disabled',
            'info'
        );
    }

    async playAudio(audioUrl) {
        try {
            const audio = new Audio(audioUrl);
            await audio.play();
        } catch (error) {
            console.error('Failed to play audio:', error);
            this.showError('Failed to play voice response');
        }
    }

    clearChat() {
        if (confirm('Are you sure you want to clear the chat history?')) {
            if (this.chatMessages) {
                this.chatMessages.innerHTML = '';
            }
            this.messageHistory = [];
            this.sessionStats = {
                messages: 0,
                responses: 0,
                avgResponseTime: 0,
                totalTime: 0
            };
            this.updateSessionStats();
            this.showWelcomeMessage();
        }
    }

    toggleStats() {
        if (!this.statsContent || !this.statsToggle) return;
        
        const isExpanded = this.statsContent.classList.contains('expanded');
        
        if (isExpanded) {
            this.statsContent.classList.remove('expanded');
            this.statsToggle.textContent = '📊';
        } else {
            this.statsContent.classList.add('expanded');
            this.statsToggle.textContent = '✕';
            this.updateSessionStats();
        }
    }

    updateSessionStats() {
        const elements = {
            messagesCount: document.getElementById('messagesCount'),
            responsesCount: document.getElementById('responsesCount'),
            avgResponseTime: document.getElementById('avgResponseTime'),
            sessionTime: document.getElementById('sessionTime')
        };

        if (elements.messagesCount) {
            elements.messagesCount.textContent = this.sessionStats.messages;
        }
        
        if (elements.responsesCount) {
            elements.responsesCount.textContent = this.sessionStats.responses;
        }
        
        if (elements.avgResponseTime) {
            elements.avgResponseTime.textContent = `${this.sessionStats.avgResponseTime}ms`;
        }
        
        if (elements.sessionTime) {
            const sessionTime = Math.floor(this.sessionStats.totalTime / 1000);
            const minutes = Math.floor(sessionTime / 60);
            const seconds = sessionTime % 60;
            elements.sessionTime.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }
    }

    updateResponseTime(responseTime) {
        if (this.sessionStats.responses === 0) {
            this.sessionStats.avgResponseTime = responseTime;
        } else {
            this.sessionStats.avgResponseTime = Math.round(
                (this.sessionStats.avgResponseTime * this.sessionStats.responses + responseTime) /
                (this.sessionStats.responses + 1)
            );
        }
    }

    loadSessionStats() {
        try {
            const stored = localStorage.getItem('housegpt_session_stats');
            if (stored) {
                this.sessionStats = { ...this.sessionStats, ...JSON.parse(stored) };
            }
        } catch (error) {
            console.error('Failed to load session stats:', error);
        }
    }

    saveSessionStats() {
        try {
            localStorage.setItem('housegpt_session_stats', JSON.stringify(this.sessionStats));
        } catch (error) {
            console.error('Failed to save session stats:', error);
        }
    }

    getSessionId() {
        let sessionId = localStorage.getItem('housegpt_session_id');
        if (!sessionId) {
            sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('housegpt_session_id', sessionId);
        }
        return sessionId;
    }

    showError(message) {
        this.showToast(message, 'error');
    }

    showToast(message, type = 'info') {
        // Remove existing toasts
        document.querySelectorAll('.toast').forEach(toast => toast.remove());

        const toast = document.createElement('div');
        toast.className = `error-toast toast ${type}`;
        
        const icon = type === 'error' ? '❌' : type === 'success' ? '✅' : 'ℹ️';
        
        toast.innerHTML = `
            <div class="error-content">
                <span class="error-icon">${icon}</span>
                <span class="error-message">${message}</span>
                <button class="error-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;

        document.body.appendChild(toast);
        
        // Show toast
        setTimeout(() => toast.classList.add('show'), 100);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 5000);
    }

    // Periodic connection check
    startConnectionMonitoring() {
        setInterval(() => this.checkConnection(), 30000); // Check every 30 seconds
    }

    // Session time tracking
    startSessionTimer() {
        const startTime = Date.now();
        setInterval(() => {
            this.sessionStats.totalTime = Date.now() - startTime;
            this.saveSessionStats();
        }, 1000);
    }
}

// Export for global access
window.HouseGPTChat = HouseGPTChat;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.chatApp = new HouseGPTChat();
    window.chatApp.startConnectionMonitoring();
    window.chatApp.startSessionTimer();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && window.chatApp) {
        window.chatApp.checkConnection();
    }
});

// Handle browser back/forward
window.addEventListener('popstate', () => {
    if (window.chatApp) {
        window.chatApp.checkConnection();
    }
});

// Global error handler
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    if (window.chatApp) {
        window.chatApp.showError('An unexpected error occurred');
    }
});

// Handle unhandled promise rejections
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    if (window.chatApp) {
        window.chatApp.showError('A network error occurred');
    }
});
