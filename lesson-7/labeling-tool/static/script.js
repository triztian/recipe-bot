document.addEventListener('DOMContentLoaded', () => {
    const traceView = document.getElementById('trace-view');
    const feedbackBox = document.getElementById('feedback-box');
    const failureModesDropdown = document.getElementById('failure-modes-dropdown');
    const newFailureModeInput = document.getElementById('new-failure-mode-input');
    const addFailureModeBtn = document.getElementById('add-failure-mode-btn');
    const progressBar = document.createElement('div');
    progressBar.id = 'progress-bar-inner';
    document.getElementById('progress-bar').appendChild(progressBar);
    const progressText = document.getElementById('progress-text');

    let traces = [];
    let labeledData = {};
    let currentTraceIndex = 0;
    let failureModes = new Set();

    async function fetchTraces() {
        try {
            const response = await fetch('/api/traces');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            traces = await response.json();
            if (traces.length > 0) {
                renderTrace(currentTraceIndex);
                updateProgress();
            } else {
                traceView.innerHTML = '<p>No traces found.</p>';
            }
        } catch (error) {
            console.error("Could not fetch traces:", error);
            traceView.innerHTML = `<p>Error loading traces: ${error.message}. Please make sure the server is running and traces.csv is in the correct location.</p>`;
        }
    }

    function renderTrace(index) {
        const trace = traces[index];
        traceView.innerHTML = ''; // Clear previous trace

        const persona = document.createElement('div');
        persona.className = 'message';
        persona.innerHTML = `<div class="sender">CUSTOMER PERSONA</div><div>${trace.customer_persona}</div>`;
        traceView.appendChild(persona);

        const query = document.createElement('div');
        query.className = 'message';
        query.innerHTML = `<div class="sender">USER QUERY</div><div>${trace.user_query}</div>`;
        traceView.appendChild(query);
        
        const conversationMessages = trace.conversation_messages.split('|').map(s => s.trim());
        conversationMessages.forEach(msg => {
            const msgDiv = document.createElement('div');
            msgDiv.className = 'message';
            if (msg.startsWith('USER:')) {
                msgDiv.classList.add('user');
                msgDiv.innerHTML = `<div class="sender">USER</div><div>${msg.substring(5)}</div>`;
            } else if (msg.startsWith('AGENT:')) {
                msgDiv.classList.add('agent');
                msgDiv.innerHTML = `<div class="sender">AGENT</div><div>${msg.substring(7)}</div>`;
            } else if (msg.startsWith('TOOL (')) {
                msgDiv.classList.add('tool');
                msgDiv.innerHTML = `<div class="sender">TOOL</div><code>${msg}</code>`;
            }
            traceView.appendChild(msgDiv);
        });

        if(trace.tool_calls) {
            const toolDiv = document.createElement('div');
            toolDiv.className = 'message tool';
            toolDiv.innerHTML = `<div class="sender">TOOL CALLS</div><code>${trace.tool_calls}</code>`;
            traceView.appendChild(toolDiv);
        }
        
        // Restore saved label if exists
        const savedLabel = labeledData[trace.trace_id];
        if (savedLabel) {
            feedbackBox.value = savedLabel.feedback;
            Array.from(failureModesDropdown.options).forEach(option => {
                option.selected = savedLabel.failure_modes.includes(option.value);
            });
        } else {
            feedbackBox.value = '';
            Array.from(failureModesDropdown.options).forEach(option => option.selected = false);
        }

        traceView.scrollTop = 0;
        updateProgress();
    }

    function updateProgress() {
        const labeledCount = Object.keys(labeledData).length;
        const totalCount = traces.length;
        progressText.textContent = `${labeledCount}/${totalCount}`;
        progressBar.style.width = totalCount > 0 ? `${(labeledCount / totalCount) * 100}%` : '0%';
    }
    
    async function saveCurrentLabel() {
        if (traces.length === 0) return;
        const trace = traces[currentTraceIndex];
        const selectedFailureModes = Array.from(failureModesDropdown.selectedOptions).map(opt => opt.value);

        const label = {
            trace_id: trace.trace_id,
            feedback: feedbackBox.value,
            failure_modes: selectedFailureModes
        };

        labeledData[trace.trace_id] = label;

        try {
            await fetch('/api/label', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(label),
            });
        } catch (error) {
            console.error("Could not save label:", error);
            // Handle save error, maybe show a notification to the user
        }
        updateProgress();
    }

    function navigate(direction) {
        saveCurrentLabel().then(() => {
            let nextIndex = currentTraceIndex + direction;
            if (nextIndex >= 0 && nextIndex < traces.length) {
                currentTraceIndex = nextIndex;
                renderTrace(currentTraceIndex);
            }
        });
    }

    addFailureModeBtn.addEventListener('click', () => {
        const newMode = newFailureModeInput.value.trim();
        if (newMode && !failureModes.has(newMode)) {
            failureModes.add(newMode);
            const option = document.createElement('option');
            option.value = newMode;
            option.textContent = newMode;
            failureModesDropdown.appendChild(option);
            newFailureModeInput.value = '';
        }
    });

    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') {
            return;
        }
        if (e.key === 'ArrowRight') {
            navigate(1);
        } else if (e.key === 'ArrowLeft') {
            navigate(-1);
        }
    });

    fetchTraces();
});
