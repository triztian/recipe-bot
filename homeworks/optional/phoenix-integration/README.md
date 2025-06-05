# Homework: Phoenix Tracing Integration for Recipe Chatbot

> This homework is best completed after "Lesson 6: Production Monitoring & Continuous Evaluation

## Overview

**This assignment focuses on implementing observability and tracing for your Recipe Chatbot using Arize Phoenix.** Phoenix provides real-time monitoring, debugging, and evaluation capabilities for LLM applications. You'll learn to instrument your chatbot, visualize conversation flows, and gain insights into your application's behavior.

This homework teaches you to add comprehensive tracing and observability to conversational AI systems.

## Learning Objectives

By completing this assignment, you will:
- Set up Phoenix tracing infrastructure for LLM applications
- Implement session-based conversation tracking
- Instrument FastAPI endpoints and LiteLLM calls
- Visualize conversation flows and debug chat interactions
- Monitor application performance and user behavior patterns

## Video Overview

Please watch [this homework video](https://www.loom.com/share/30e5dced1afa467e8714d515694bb3b0) and then complete the steps below to implement it yourself.

## Part 1: Phoenix Infrastructure Setup

### Step 1: Install Phoenix Dependencies

Add Phoenix packages to your `requirements.txt`:

```
arize-phoenix-otel
openinference-instrumentation-litellm
openinference-instrumentation
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

### Step 2: Launch Phoenix Server

Phoenix requires a running server to collect and visualize traces. Start the Phoenix server:

```bash
# In a terminal window (keep this running)
phoenix serve
```

This starts Phoenix at `http://localhost:6006` where you can view traces and conversations.

## Part 2: Implement Phoenix Tracing

### Step 3: Session Management in Frontend

First, let's implement proper session ID generation and management in the frontend, since the backend will need to receive these session IDs.

Modify your `frontend/index.html` to properly generate and maintain session IDs:

> Note:  This is javascript, so it must be inside a `<script>` tag.

```javascript
let sessionId = generateSessionId(); // Generate initial session ID

/**
 * Generate a unique session identifier.
 */
function generateSessionId() {
    // TODO: Create a unique session ID using timestamp and random string
    // Format: 'session_' + timestamp + '_' + random_string
}

async function sendMessage(evt) {
    // ... existing code ...
    
    // In the fetch to the backend `/chat` route, send session ID in addition to `chatHistory`

    const res = await fetch("/chat", {
            // TODO: Ensure session ID is included in the body
        }),
    
        
        const data = await res.json();
        chatHistory = data.messages;
        // TODO: set session ID from server response similar to how `ChatHistory` is set
    // ... existing code ...
}
```

**Your Task**: Complete the TODOs by:
1. Implementing `generateSessionId()` to create unique identifiers using timestamp and random string
2. Ensuring the session ID is included in every chat request
3. Updating the local session ID with the response from the server

### Step 4: Create Tracing Configuration

Create `backend/tracing.py` to configure Phoenix tracing:

```python
"""Phoenix tracing configuration for the recipe chatbot."""

import os
import uuid
from phoenix.otel import register
from openinference.instrumentation.litellm import LiteLLMInstrumentor

def setup_local_phoenix_tracing():
    """Initialize Phoenix tracing with environment configuration."""
    # If you are self-hosting or using cloud you will also need an API key
    # For this assignment you do not need that
    phoenix_endpoint = "http://0.0.0.0:6006"
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = phoenix_endpoint
    os.environ["PHOENIX_ENDPOINT"] = phoenix_endpoint    
    print(f"Using local Phoenix endpoint: {phoenix_endpoint}")
    
    # TODO: Instrument LiteLLM before registering Phoenix
    
    # TODO: Configure the Phoenix tracer with project name
    
    print("Phoenix tracing initialized successfully")
    # TODO: Return tracer for use in application
```

**Your Task**: Complete the TODOs in this function by:
1. Copy starter code into `backend/tracing.py` file you create
2. Use `LiteLLMInstrumentor` to instrument LiteLLM calls
3. Call `register()` with appropriate project name and auto-instrumentation
4. Returning a tracer from the tracer provider

### Step 5: Integrate Tracing in Main Application

Now that the frontend is sending session IDs, modify your `backend/main.py` to integrate Phoenix tracing and use the session IDs:

```python
from openinference.instrumentation import using_session
from openinference.semconv.trace import SpanAttributes
from backend.tracing import setup_local_phoenix_tracing

# TODO: Initialize Phoenix tracing and get tracer

# TODO: Add session_id field to the `ChatRequest` and `ChatResponse` object

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """Main conversational endpoint with Phoenix tracing."""
    # TODO: Get session ID from the payload or create a random one if there isn't one
    session_id = payload.session_id or str(uuid.uuid4())
    
    # ... existing code

    try:
        # TODO: Create a span named "chat_turn" with agent span kind (context manager)
        # TODO: Set session ID and input/output attributes on the span
        # TODO: Use session context to propagate session ID to child spans
        # TODO: Call get_agent_response within the tracing and using session context
        
        updated_messages_dicts = get_agent_response(request_messages)
        
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    # ... existing code

    # TODO: Add session ID to the chat response being returned
```

**Your Task**: Complete the TODOs by:
1. Setting up Phoenix tracing with `setup_local_phoenix_tracing()`
2. Creating spans with proper attributes (session ID, input/output values, span kind)
3. Using `using_session()` context manager for session correlation
4. Handling span lifecycle (start, set attributes, end)

## Part 3: Testing and Validation

### Step 6: Test Your Implementation

1. **Start Phoenix Server**:
   ```bash
   phoenix serve
   ```

2. **Run Your Chatbot**:
   ```bash
   uvicorn backend.main:app --reload
   ```

3. **Test Conversations**:
   - Open `http://localhost:8000` in your browser
   - Have several conversations with the chatbot
   - Try different types of recipe requests

4. **Verify Tracing**:
   - Open Phoenix UI at `http://localhost:6006`
   - Navigate to "Spans", "Traces", and "Sessions" tabs and verify proper groupings

5. **Next Steps**
   - Try annotating in pheonix UI
   - Create a dataset and add some traces to that dataset
