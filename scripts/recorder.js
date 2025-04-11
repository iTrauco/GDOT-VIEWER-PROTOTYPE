// Recorder functionality for GDOT Stream Viewer

// Elements
const recordButton = document.getElementById('recordBtn');
const recordStatus = document.getElementById('recordStatus');
const recordingsList = document.getElementById('recordingsList');

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
  // Setup event listeners
  recordButton.addEventListener('click', startRecording);
  
  // Check for existing recordings
  fetchRecordings();
});

/**
 * Start the recording process via AJAX request to the server
 */
function startRecording() {
  const url = document.getElementById('streamUrl').value;
  
  if (!url) {
    updateStatus('Please enter a stream URL first', 'error');
    return;
  }
  
  // Disable the record button while recording
  recordButton.disabled = true;
  recordStatus.textContent = "Recording... (30s)";
  
  // Make AJAX request to the server to start recording
  fetch('/record', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      url: url,
      duration: 30
    }),
  })
  .then(response => response.json())
  .then(data => {
    if (data.success) {
      // Start countdown timer
      let timeLeft = 30;
      const countdownInterval = setInterval(() => {
        timeLeft--;
        recordStatus.textContent = `Recording... (${timeLeft}s)`;
        
        if (timeLeft <= 0) {
          clearInterval(countdownInterval);
          recordButton.disabled = false;
          recordStatus.textContent = "Recording complete";
          
          // Check for new recordings
          setTimeout(() => {
            fetchRecordings();
            recordStatus.textContent = "";
          }, 3000);
        }
      }, 1000);
    } else {
      recordButton.disabled = false;
      recordStatus.textContent = "Recording failed";
      updateStatus(`Recording failed: ${data.error}`, 'error');
    }
  })
  .catch(error => {
    console.error('Error:', error);
    recordButton.disabled = false;
    recordStatus.textContent = "Recording failed";
    updateStatus('Error connecting to recording server', 'error');
  });
}

/**
 * Fetch the list of existing recordings from the server
 */
function fetchRecordings() {
  fetch('/recordings')
    .then(response => response.json())
    .then(data => {
      if (data.success && data.recordings && data.recordings.length > 0) {
        updateRecordingsList(data.recordings);
      } else {
        recordingsList.innerHTML = '<p>No recordings found</p>';
      }
    })
    .catch(error => {
      console.error('Error fetching recordings:', error);
      recordingsList.innerHTML = '<p>Error loading recordings</p>';
    });
}

/**
 * Update the recordings list in the UI
 * @param {Array} recordings - List of recording objects
 */
function updateRecordingsList(recordings) {
  // Clear the current list
  recordingsList.innerHTML = '';
  
  // Add each recording to the list
  recordings.forEach(recording => {
    const item = document.createElement('div');
    item.className = 'recording-item';
    
    const link = document.createElement('a');
    link.href = `/recordings/${recording.filename}`;
    link.textContent = recording.name || recording.filename;
    link.target = '_blank';
    
    const timestamp = document.createElement('span');
    timestamp.className = 'timestamp';
    timestamp.textContent = recording.timestamp || '';
    
    item.appendChild(link);
    item.appendChild(timestamp);
    recordingsList.appendChild(item);
  });
}