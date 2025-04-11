// Recorder functionality for GDOT Stream Viewer

// Elements
const recordButton = document.getElementById('recordBtn');
const recordStatus = document.getElementById('recordStatus');
const recordingsList = document.getElementById('recordingsList');
const recordingStats = document.getElementById('recordingStats');
const resolutionStat = document.getElementById('resolutionStat');
const fpsStat = document.getElementById('fpsStat');
const fileSizeStat = document.getElementById('fileSizeStat');

// Stats polling interval
let statsInterval = null;

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
  
  // Reset and show stats display
  resolutionStat.textContent = "-";
  fpsStat.textContent = "-";
  fileSizeStat.textContent = "0 B";
  recordingStats.className = "recording-stats active";
  
  // Start stats polling
  startStatsPolling();
  
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
          
          // Stop stats polling
          stopStatsPolling();
          
          // Hide stats display after a delay
          setTimeout(() => {
            recordingStats.className = "recording-stats";
          }, 3000);
          
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
      stopStatsPolling();
      recordingStats.className = "recording-stats";
      updateStatus(`Recording failed: ${data.error}`, 'error');
    }
  })
  .catch(error => {
    console.error('Error:', error);
    recordButton.disabled = false;
    recordStatus.textContent = "Recording failed";
    stopStatsPolling();
    recordingStats.className = "recording-stats";
    updateStatus('Error connecting to recording server', 'error');
  });
}

/**
 * Start polling for recording stats
 */
function startStatsPolling() {
  console.log("Starting stats polling...");
  
  // Clear any existing interval
  if (statsInterval) {
    clearInterval(statsInterval);
  }
  
  // Poll every second
  statsInterval = setInterval(() => {
    console.log("Polling for stats...");
    fetch('/recording-stats')
      .then(response => response.json())
      .then(data => {
        console.log("Received stats:", data);
        if (data.active) {
          resolutionStat.textContent = data.resolution;
          fpsStat.textContent = data.fps;
          fileSizeStat.textContent = data.file_size;
        } else {
          // If recording is not active but we're still polling, stop
          if (!recordButton.disabled) {
            stopStatsPolling();
            recordingStats.className = "recording-stats";
          }
        }
      })
      .catch(error => {
        console.error('Error fetching stats:', error);
      });
  }, 1000);
}

/**
 * Stop polling for recording stats
 */
function stopStatsPolling() {
  if (statsInterval) {
    clearInterval(statsInterval);
    statsInterval = null;
  }
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