// Viewer functionality for GDOT Stream Viewer

// Elements
const videoPlayer = document.getElementById('videoPlayer');
const streamUrl = document.getElementById('streamUrl');
const loadBtn = document.getElementById('loadBtn');
const statusBox = document.getElementById('status');

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
  // Setup event listeners
  loadBtn.addEventListener('click', loadStream);
  
  // Try to load the default stream on page load
  setTimeout(loadStream, 500);
});

/**
 * Load and play the HLS stream
 */
function loadStream() {
  const url = streamUrl.value;
  
  if (!url) {
    updateStatus('Please enter a stream URL', 'error');
    return;
  }
  
  updateStatus('Loading stream...', 'info');
  
  // Stop any current stream
  if (window.hls) {
    window.hls.destroy();
    window.hls = null;
  }
  
  // Check if HLS.js is supported
  if (Hls.isSupported()) {
    window.hls = new Hls({
      debug: false,
      enableWorker: true,
      lowLatencyMode: true,
      backBufferLength: 90
    });
    
    window.hls.loadSource(url);
    window.hls.attachMedia(videoPlayer);
    
    window.hls.on(Hls.Events.MANIFEST_PARSED, function() {
      updateStatus('Stream loaded successfully', 'success');
      videoPlayer.play()
        .catch(e => {
          console.error('Error playing video:', e);
          updateStatus('Stream loaded, but autoplay failed. Press play to start.', 'info');
        });
    });
    
    window.hls.on(Hls.Events.ERROR, function(event, data) {
      console.error('HLS Error:', data);
      if (data.fatal) {
        updateStatus(`Stream error: ${data.details}`, 'error');
        window.hls.destroy();
      }
    });
  } 
  // For browsers with native HLS support (like Safari)
  else if (videoPlayer.canPlayType('application/vnd.apple.mpegurl')) {
    videoPlayer.src = url;
    videoPlayer.addEventListener('loadedmetadata', function() {
      updateStatus('Stream loaded successfully (native HLS)', 'success');
      videoPlayer.play()
        .catch(e => {
          console.error('Error playing video:', e);
          updateStatus('Stream loaded, but autoplay failed. Press play to start.', 'info');
        });
    });
    
    videoPlayer.addEventListener('error', function() {
      updateStatus('Error loading stream with native support', 'error');
    });
  } 
  else {
    updateStatus('HLS is not supported in your browser', 'error');
  }
}

/**
 * Update the status box with a message
 * @param {string} message - The message to display
 * @param {string} type - The type of message (success, error, info)
 */
function updateStatus(message, type = 'info') {
  statusBox.textContent = message;
  statusBox.className = 'status-box ' + type;
  
  // Auto clear success messages after 5 seconds
  if (type === 'success') {
    setTimeout(() => {
      statusBox.textContent = '';
      statusBox.className = 'status-box';
    }, 5000);
  }
}