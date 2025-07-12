"""
Demo Interface

Simple web interface for the chatbot demo.
"""


def get_chat_html() -> str:
    """Return the HTML for the chat interface."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Simple Chatbot Demo</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }
            
            .header {
                background: rgba(255, 255, 255, 0.1);
                padding: 1rem;
                text-align: center;
                backdrop-filter: blur(10px);
                border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .header h1 {
                color: white;
                font-size: 1.8rem;
                margin-bottom: 0.5rem;
            }
            
            .header p {
                color: rgba(255, 255, 255, 0.8);
                font-size: 0.9rem;
            }
            
            .chat-container {
                flex: 1;
                display: flex;
                flex-direction: column;
                max-width: 800px;
                margin: 0 auto;
                padding: 1rem;
                width: 100%;
            }
            
            .chat-messages {
                flex: 1;
                background: rgba(255, 255, 255, 0.95);
                border-radius: 10px;
                padding: 1rem;
                margin-bottom: 1rem;
                overflow-y: auto;
                max-height: 60vh;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .message {
                margin-bottom: 1rem;
                padding: 0.75rem;
                border-radius: 8px;
                max-width: 80%;
            }
            
            .user-message {
                background: #e3f2fd;
                margin-left: auto;
                text-align: right;
            }
            
            .bot-message {
                background: #f5f5f5;
                margin-right: auto;
            }
            
            .message-content {
                line-height: 1.4;
                white-space: pre-wrap;
            }
            
            .input-container {
                display: flex;
                gap: 0.5rem;
                background: rgba(255, 255, 255, 0.95);
                padding: 1rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .input-field {
                flex: 1;
                padding: 0.75rem;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 1rem;
                resize: none;
                min-height: 40px;
                max-height: 120px;
            }
            
            .send-button {
                padding: 0.75rem 1.5rem;
                background: #0984e3;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 1rem;
                transition: background 0.3s;
                white-space: nowrap;
            }
            
            .send-button:hover:not(:disabled) {
                background: #0672c7;
            }
            
            .send-button:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            
            .clear-button {
                padding: 0.5rem 1rem;
                background: #dc3545;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 0.9rem;
                transition: background 0.3s;
                margin-left: 0.5rem;
            }
            
            .clear-button:hover:not(:disabled) {
                background: #c82333;
            }
            
            .clear-button:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            
            .loading {
                display: none;
                text-align: center;
                color: #666;
                font-style: italic;
                padding: 1rem;
            }
            
            .welcome-message {
                text-align: center;
                color: #666;
                font-style: italic;
                padding: 2rem;
            }
            
            .examples {
                background: rgba(255, 255, 255, 0.9);
                border-radius: 10px;
                padding: 1rem;
                margin-bottom: 1rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .examples h3 {
                color: #333;
                margin-bottom: 0.5rem;
                font-size: 1rem;
            }
            
            .examples ul {
                list-style: none;
                padding: 0;
            }
            
            .examples li {
                margin: 0.3rem 0;
                padding: 0.5rem;
                background: #f8f9fa;
                border-radius: 5px;
                cursor: pointer;
                transition: background 0.3s;
                font-size: 0.9rem;
            }
            
            .examples li:hover {
                background: #e9ecef;
            }
            
            @media (max-width: 768px) {
                .chat-container {
                    padding: 0.5rem;
                }
                
                .message {
                    max-width: 90%;
                }
                
                .header h1 {
                    font-size: 1.5rem;
                }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🤖 Simple Chatbot Demo</h1>
            <p>A basic conversational AI assistant powered by LangGraph</p>
        </div>
        
        <div class="chat-container">
            <div class="examples">
                <h3>💡 Try asking:</h3>
                <ul>
                    <li onclick="askExample(this)">What is machine learning?</li>
                    <li onclick="askExample(this)">My name is John, nice to meet you!</li>
                    <li onclick="askExample(this)">Tell me a short story about a robot</li>
                    <li onclick="askExample(this)">What did we just talk about?</li>
                </ul>
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="welcome-message">
                    👋 Hello! I'm a simple AI assistant with conversation memory. I can remember our chat history and refer back to previous messages!
                </div>
            </div>
            
            <div class="loading" id="loading">
                🤔 Thinking...
            </div>
            
            <div class="input-container">
                <textarea 
                    id="messageInput" 
                    class="input-field" 
                    placeholder="Type your message here..."
                    rows="1"
                    onkeypress="handleKeyPress(event)"
                    oninput="autoResize(this)"
                ></textarea>
                <button onclick="sendMessage()" id="sendButton" class="send-button">Send</button>
                <button onclick="clearConversation()" id="clearButton" class="clear-button">Clear</button>
            </div>
        </div>
        
        <script>
            function askExample(element) {
                document.getElementById('messageInput').value = element.textContent;
                sendMessage();
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    sendMessage();
                }
            }
            
            function autoResize(textarea) {
                textarea.style.height = 'auto';
                textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
            }
            
            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                
                if (!message) return;
                
                const messagesContainer = document.getElementById('chatMessages');
                const loading = document.getElementById('loading');
                const sendButton = document.getElementById('sendButton');
                
                // Clear welcome message if present
                const welcomeMessage = messagesContainer.querySelector('.welcome-message');
                if (welcomeMessage) {
                    welcomeMessage.remove();
                }
                
                // Add user message
                const userMessage = document.createElement('div');
                userMessage.className = 'message user-message';
                userMessage.innerHTML = `<div class="message-content">${message}</div>`;
                messagesContainer.appendChild(userMessage);
                
                // Clear input and disable button
                input.value = '';
                input.style.height = 'auto';
                sendButton.disabled = true;
                loading.style.display = 'block';
                
                // Scroll to bottom
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ message: message })
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    
                    // Add bot response
                    const botMessage = document.createElement('div');
                    botMessage.className = 'message bot-message';
                    botMessage.innerHTML = `<div class="message-content">${data.response}</div>`;
                    messagesContainer.appendChild(botMessage);
                    
                } catch (error) {
                    console.error('Error:', error);
                    const errorMessage = document.createElement('div');
                    errorMessage.className = 'message bot-message';
                    errorMessage.innerHTML = `<div class="message-content">❌ Sorry, I encountered an error: ${error.message}</div>`;
                    messagesContainer.appendChild(errorMessage);
                } finally {
                    loading.style.display = 'none';
                    sendButton.disabled = false;
                    messagesContainer.scrollTop = messagesContainer.scrollHeight;
                    input.focus();
                }
            }
            
            async function clearConversation() {
                const clearButton = document.getElementById('clearButton');
                const messagesContainer = document.getElementById('chatMessages');
                
                if (!confirm('Are you sure you want to clear the conversation?')) {
                    return;
                }
                
                clearButton.disabled = true;
                
                try {
                    const response = await fetch('/clear', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        }
                    });
                    
                    if (response.ok) {
                        // Clear the UI
                        messagesContainer.innerHTML = '<div class="welcome-message">👋 Conversation cleared! Ask me anything!</div>';
                    } else {
                        console.error('Failed to clear conversation');
                    }
                } catch (error) {
                    console.error('Error clearing conversation:', error);
                } finally {
                    clearButton.disabled = false;
                }
            }
            
            // Focus on input when page loads
            window.addEventListener('load', () => {
                document.getElementById('messageInput').focus();
            });
        </script>
    </body>
    </html>
    """
