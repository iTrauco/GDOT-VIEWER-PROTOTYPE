# GDOT Traffic Camera Extraction and Recording Project

## Project Objective

Create an automated system to access and record video streams from the Georgia Department of Transportation (GDOT) traffic cameras without using browser automation.

## Technical Discovery

After analyzing the GDOT traffic camera system, we discovered that:

1. The website (511ga.org) displays hundreds of traffic cameras across Georgia
2. Each camera has an ID visible in the UI (format: "ATL-0092" or "COBB-0883")
3. The actual video streams follow a predictable URL pattern:
   ```
   https://sfs-msc-pub-lq-01.navigator.dot.ga.gov:443/rtplive/[PREFIX]-CCTV-[NUMBER]/playlist.m3u8
   ```
4. For example, camera "ATL-0092" uses the stream URL:
   ```
   https://sfs-msc-pub-lq-01.navigator.dot.ga.gov:443/rtplive/ATL-CCTV-0092/playlist.m3u8
   ```

## Implementation Approach

Rather than using Selenium or Robot Framework to navigate the website, click buttons, and handle popups, we can directly access these streams using:

1. A conversion function that transforms camera IDs into stream URLs
2. FFmpeg to connect to and record from these HLS streams
3. A simple script to process multiple cameras efficiently

## Current Solution

We've created a Python script that:
- Converts camera IDs to stream URLs using the discovered pattern
- Uses FFmpeg to record from each stream for a configurable duration
- Saves recordings with timestamped filenames
- Requires no browser interaction

## Key Components

1. **ID to URL Conversion**:
   ```python
   def convert_id_to_url(camera_id):
       # Split the ID to get the prefix (ATL, COBB, etc.) and the number
       parts = camera_id.split('-')
       prefix = parts[0]  # ATL, COBB, etc.
       number = parts[1]  # The numeric part
       
       # Format: PREFIX-CCTV-NUMBER
       formatted_id = f"{prefix}-CCTV-{number}"
       full_url = f"{BASE_URL}{formatted_id}{SUFFIX}"
       
       return full_url
   ```

2. **FFmpeg Integration**:
   ```python
   cmd = [
       "ffmpeg", 
       "-y",                 # Overwrite output files
       "-i", url,            # Input URL
       "-t", str(duration),  # Duration in seconds
       "-c", "copy",         # Copy streams without re-encoding
       "-bsf:a", "aac_adtstoasc",  # Fix for HLS streams
       output_file
   ]
   ```

## Expansion Plans

1. **Mass Camera Collection**:
   - Create a comprehensive list of all camera IDs (estimated 300+ cameras)
   - Implement batch processing with configurable limits

2. **Scheduling and Monitoring**:
   - Set up automated recording at specific times or intervals
   - Implement health checks and notification systems

3. **Data Management**:
   - Organize recordings by geography, timestamp, or other metadata
   - Implement retention policies for storage management

## Technical Requirements

- Python 3
- FFmpeg (installed and in system PATH)
- Sufficient disk space for video storage
- Internet connection with access to GDOT servers

## Current Status

The direct access approach has been tested with sample cameras and works successfully. No browser or website interaction is needed to access and record these streams.

The solution is ready to be expanded to handle all available cameras in the GDOT network.
