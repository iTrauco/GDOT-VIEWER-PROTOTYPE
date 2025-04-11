# GDOT Traffic Camera Streaming and Recording: Implementation Details

## Overview

This document outlines the implementation details for accessing, viewing, and recording Georgia Department of Transportation (GDOT) traffic camera streams without having to navigate to the official website. The solution bypasses the need for web scraping by directly accessing the HLS (HTTP Live Streaming) stream URLs.

## Stream URL Pattern

GDOT traffic cameras follow a consistent URL pattern:

```
https://sfs-msc-pub-lq-01.navigator.dot.ga.gov:443/rtplive/PREFIX-CCTV-NUMBER/playlist.m3u8
```

For example:
- ATL-0092: `https://sfs-msc-pub-lq-01.navigator.dot.ga.gov:443/rtplive/ATL-CCTV-0092/playlist.m3u8`
- ATL-0093: `https://sfs-msc-pub-lq-01.navigator.dot.ga.gov:443/rtplive/ATL-CCTV-0093/playlist.m3u8`
- ATL-0610: `https://sfs-msc-pub-lq-01.navigator.dot.ga.gov:443/rtplive/ATL-CCTV-0610/playlist.m3u8`

## Architecture

The implementation consists of:

1. **Web Interface**: HTML/JavaScript frontend for viewing streams using HLS.js
2. **Python Server**: Simple HTTP server with recording capabilities
3. **Recording Scripts**: Methods to capture stream segments using curl and FFmpeg

## Technical Challenges

### Challenge 1: FFmpeg SSL Support

**Problem**: Many FFmpeg installations lack SSL support, preventing direct downloading of HTTPS streams.

**Solution**: Multi-step approach:
1. Use curl (which has SSL support) to download the playlist and chunklist files
2. Use curl to download the individual video segments locally
3. Use FFmpeg to process local files instead of HTTPS streams

### Challenge 2: HLS Stream Structure

**Problem**: HLS streams use a multi-level playlist structure.

**Solution**: Three-stage processing:
1. Download main playlist (playlist.m3u8)
2. Extract and download chunklist URL
3. Download and process individual media segments (.ts files)

## Implementation Details

### Viewing Streams

For viewing, we use HLS.js which handles all the complexities of HLS streaming:

```javascript
// JavaScript for loading the stream
if(Hls.isSupported()) {
  const hls = new Hls();
  hls.loadSource(url);  // The GDOT camera URL
  hls.attachMedia(videoElement);
}
```

### Recording Streams

Recording is more complex due to the FFmpeg SSL issue. Here's the workflow:

1. **Download Main Playlist**:
   ```python
   playlist_file = os.path.join(temp_dir, 'playlist.m3u8')
   subprocess.run(['curl', '-s', url, '-o', playlist_file], check=True)
   ```

2. **Extract Chunklist URL**:
   ```python
   with open(playlist_file, 'r') as f:
       playlist_content = f.read()
   chunklist_match = re.search(r'(chunklist.*\.m3u8)', playlist_content)
   chunklist = chunklist_match.group(1)
   chunklist_url = f"{base_url}/{chunklist}"
   ```

3. **Download Chunklist**:
   ```python
   chunklist_file = os.path.join(temp_dir, 'chunklist.m3u8')
   subprocess.run(['curl', '-s', chunklist_url, '-o', chunklist_file], check=True)
   ```

4. **Download Video Segments**:
   ```python
   segments = re.findall(r'media_\w+\.ts', chunklist_content)
   local_segments_dir = os.path.join(temp_dir, 'segments')
   os.makedirs(local_segments_dir, exist_ok=True)
   
   for segment in segments:
       segment_url = f"{segment_base_url}/{segment}"
       local_segment = os.path.join(local_segments_dir, segment)
       subprocess.run(['curl', '-s', segment_url, '-o', local_segment], check=True)
   ```

5. **Modify Chunklist to Use Local Paths**:
   ```python
   modified_content = re.sub(r'media_(\w+\.ts)', f'segments/media_\\1', chunklist_content)
   with open(chunklist_file, 'w') as f:
       f.write(modified_content)
   ```

6. **Process with FFmpeg**:
   ```python
   subprocess.run([
       'ffmpeg',
       '-y',
       '-protocol_whitelist', 'file,http,https,tcp,tls',
       '-i', chunklist_file,
       '-t', str(duration),
       '-c', 'copy',
       '-movflags', '+faststart',
       '-avoid_negative_ts', 'make_zero',
       output_file
   ], check=True)
   ```

## Server Implementation

The server handles:
1. Serving the web interface
2. Processing recording requests
3. Managing recording files
4. Providing a list of recorded videos

It uses Python's `http.server` and `socketserver` modules for simplicity.

## Future Enhancements

1. **Live Preview Alongside Recording**: 
   - Use WebSockets to provide real-time status updates during recording
   - Implement picture-in-picture for monitoring recording progress

2. **Multi-Camera Dashboard**:
   - Support viewing and recording multiple cameras simultaneously
   - Implement a grid layout for monitoring multiple streams

3. **Scheduled Recording**:
   - Add functionality for time-based recording schedules
   - Implement motion detection to only record when activity is detected

4. **Cloud Integration**:
   - Add support for automatic uploading to cloud storage
   - Implement remote access to view and control cameras

## Requirements

- Python 3.6 or higher
- FFmpeg (with or without SSL support)
- curl
- Modern web browser with JavaScript support

## Working Example

The current implementation successfully views and records GDOT traffic camera streams without loading the official website. It handles FFmpeg installations with or without SSL support through the curl-based workaround.

The result is an MP4 file containing a 30-second clip of the traffic camera stream, saved with a timestamp in the filename.
