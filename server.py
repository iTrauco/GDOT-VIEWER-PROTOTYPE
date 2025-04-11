#!/usr/bin/env python3
"""
Simple HTTP server for GDOT Stream Viewer with recording capabilities
"""

import http.server
import socketserver
import os
import json
import subprocess
import threading
import time
import re
import shutil
from datetime import datetime
from urllib.parse import parse_qs, urlparse

# Configuration
PORT = 8000
RECORDINGS_DIR = "recordings"

print("Starting server...")

# Make sure recordings directory exists
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# Custom request handler
class GDOTRequestHandler(http.server.SimpleHTTPRequestHandler):
    
    # Add this class variable to track recording stats
    recording_stats = {
        "active": False,
        "output_file": "",
        "resolution": "-",
        "fps": "-",
        "file_size": "-",
        "start_time": 0
    }
    
    def do_GET(self):
        """Handle GET requests"""
        # API endpoint to get recording stats
        if self.path == '/recording-stats':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')  # For CORS
            self.end_headers()
            
            # Update file size if recording is active
            if GDOTRequestHandler.recording_stats["active"] and GDOTRequestHandler.recording_stats["output_file"]:
                if os.path.exists(GDOTRequestHandler.recording_stats["output_file"]):
                    size_bytes = os.path.getsize(GDOTRequestHandler.recording_stats["output_file"])
                    if size_bytes < 1024:
                        GDOTRequestHandler.recording_stats["file_size"] = f"{size_bytes} B"
                    elif size_bytes < 1024 * 1024:
                        GDOTRequestHandler.recording_stats["file_size"] = f"{size_bytes/1024:.1f} KB"
                    else:
                        GDOTRequestHandler.recording_stats["file_size"] = f"{size_bytes/(1024*1024):.1f} MB"
            
            # Debug output
            print(f"Sending recording stats: {GDOTRequestHandler.recording_stats}")
            
            self.wfile.write(json.dumps(GDOTRequestHandler.recording_stats).encode())
            return
        
        # API endpoint to get recordings list
        elif self.path == '/recordings':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Get list of recordings
            recordings = []
            for filename in os.listdir(RECORDINGS_DIR):
                if filename.endswith('.mp4'):
                    file_path = os.path.join(RECORDINGS_DIR, filename)
                    stat_info = os.stat(file_path)
                    recordings.append({
                        'filename': filename,
                        'name': self._format_recording_name(filename),
                        'size': stat_info.st_size,
                        'timestamp': datetime.fromtimestamp(stat_info.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    })
            
            # Sort by time, newest first
            recordings.sort(key=lambda x: x['timestamp'], reverse=True)
            
            self.wfile.write(json.dumps({
                'success': True,
                'recordings': recordings
            }).encode())
            return
        
        # Serve recordings
        elif self.path.startswith('/recordings/'):
            filename = self.path[12:]  # Remove '/recordings/' prefix
            file_path = os.path.join(RECORDINGS_DIR, filename)
            
            if os.path.exists(file_path) and os.path.isfile(file_path):
                self.send_response(200)
                self.send_header('Content-type', 'video/mp4')
                self.send_header('Content-Length', str(os.stat(file_path).st_size))
                self.send_header('Content-Disposition', f'inline; filename="{filename}"')
                self.end_headers()
                
                with open(file_path, 'rb') as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, 'File not found')
                return
        
        # Default: serve files from current directory
        return http.server.SimpleHTTPRequestHandler.do_GET(self)
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/record':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Extract parameters
            url = data.get('url')
            duration = int(data.get('duration', 30))
            
            if not url:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'success': False,
                    'error': 'Missing URL parameter'
                }).encode())
                return
            
            # Start recording in background thread
            threading.Thread(
                target=self._record_stream,
                args=(url, duration)
            ).start()
            
            # Respond immediately
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'success': True,
                'message': f'Recording started for {duration} seconds'
            }).encode())
            return
        
        self.send_error(404, 'Endpoint not found')
    
    def _record_stream(self, url, duration):
        """Record the stream using curl+ffmpeg method"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(RECORDINGS_DIR, f'gdot_stream_{timestamp}.mp4')
        
        # Reset and update recording stats
        GDOTRequestHandler.recording_stats["active"] = True
        GDOTRequestHandler.recording_stats["output_file"] = output_file
        GDOTRequestHandler.recording_stats["resolution"] = "-"
        GDOTRequestHandler.recording_stats["fps"] = "-"
        GDOTRequestHandler.recording_stats["file_size"] = "0 B"
        GDOTRequestHandler.recording_stats["start_time"] = time.time()
        
        print(f"Starting recording of {url} for {duration} seconds...")
        
        # Skip direct FFmpeg since we know it doesn't support HTTPS
        # Use curl+FFmpeg method directly
        temp_dir = os.path.join(RECORDINGS_DIR, f'temp_{timestamp}')
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Step 1: Download playlist
            playlist_file = os.path.join(temp_dir, 'playlist.m3u8')
            subprocess.run(['curl', '-s', url, '-o', playlist_file], check=True, timeout=10)
            
            if not os.path.exists(playlist_file) or os.path.getsize(playlist_file) == 0:
                print("Failed to download playlist")
                GDOTRequestHandler.recording_stats["active"] = False
                return
            
            # Step 2: Extract chunklist URL
            base_url = os.path.dirname(url)
            with open(playlist_file, 'r') as f:
                playlist_content = f.read()
            
            # Extract resolution info from the playlist
            resolution_match = re.search(r'RESOLUTION=(\d+x\d+)', playlist_content)
            if resolution_match:
                GDOTRequestHandler.recording_stats["resolution"] = resolution_match.group(1)
            
            chunklist_match = re.search(r'(chunklist.*\.m3u8)', playlist_content)
            if not chunklist_match:
                print("No chunklist found in playlist")
                GDOTRequestHandler.recording_stats["active"] = False
                return
            
            chunklist = chunklist_match.group(1)
            chunklist_url = f"{base_url}/{chunklist}"
            chunklist_file = os.path.join(temp_dir, 'chunklist.m3u8')
            
            # Step 3: Download chunklist
            subprocess.run(['curl', '-s', chunklist_url, '-o', chunklist_file], check=True, timeout=10)
            
            if not os.path.exists(chunklist_file) or os.path.getsize(chunklist_file) == 0:
                print("Failed to download chunklist")
                GDOTRequestHandler.recording_stats["active"] = False
                return
            
            # Step 4: Modify chunklist to use local paths
            segment_base_url = os.path.dirname(chunklist_url)
            with open(chunklist_file, 'r') as f:
                chunklist_content = f.read()

            # Extract fps info from chunklist
            extinf_matches = re.findall(r'#EXTINF:([\d.]+),', chunklist_content)
            if extinf_matches and len(extinf_matches) > 1:
                avg_segment_duration = sum(float(d) for d in extinf_matches) / len(extinf_matches)
                estimated_fps = 30.0 / avg_segment_duration  # Rough estimate
                GDOTRequestHandler.recording_stats["fps"] = f"{estimated_fps:.1f}"

            # Find all media segments
            segments = re.findall(r'media_\w+\.ts', chunklist_content)
            print(f"Found {len(segments)} media segments")

            # Download each segment
            local_segments_dir = os.path.join(temp_dir, 'segments')
            os.makedirs(local_segments_dir, exist_ok=True)
            
            # Calculate how many segments to download based on duration and EXTINF values
            # EXTINF values in the chunklist indicate segment duration
            if extinf_matches:
                avg_segment_duration = sum(float(d) for d in extinf_matches) / len(extinf_matches)
                segments_needed = min(len(segments), int(duration / avg_segment_duration) + 2)  # +2 for safety
            else:
                segments_needed = min(len(segments), 30)  # Reasonable default
                
            print(f"Downloading {segments_needed} segments for {duration}s recording")
            
            # Download only the segments we need
            for i, segment in enumerate(segments[:segments_needed]):
                segment_url = f"{segment_base_url}/{segment}"
                local_segment = os.path.join(local_segments_dir, segment)
                print(f"Downloading segment {i+1}/{segments_needed}: {segment}")
                subprocess.run(['curl', '-s', segment_url, '-o', local_segment], check=True, timeout=10)
                
                # Update file size stat after each segment
                if os.path.exists(local_segment):
                    total_size = 0
                    for seg_file in os.listdir(local_segments_dir):
                        total_size += os.path.getsize(os.path.join(local_segments_dir, seg_file))
                    
                    if total_size < 1024:
                        GDOTRequestHandler.recording_stats["file_size"] = f"{total_size} B"
                    elif total_size < 1024 * 1024:
                        GDOTRequestHandler.recording_stats["file_size"] = f"{total_size/1024:.1f} KB"
                    else:
                        GDOTRequestHandler.recording_stats["file_size"] = f"{total_size/(1024*1024):.1f} MB"

            # Create a local chunklist with only the segments we downloaded
            local_chunklist = chunklist_content
            for segment in segments[segments_needed:]:
                # Remove segments we didn't download from the chunklist
                local_chunklist = local_chunklist.replace(segment, '')
                
            # Update segment paths to local
            local_chunklist = re.sub(r'(media_\w+\.ts)', r'segments/\1', local_chunklist)
            
            # Write updated chunklist
            with open(chunklist_file, 'w') as f:
                f.write(local_chunklist)
            
            # Step 5: Use FFmpeg to combine segments into MP4
            subprocess.run([
                'ffmpeg',
                '-y',
                '-protocol_whitelist', 'file',
                '-i', chunklist_file,
                '-c', 'copy',
                '-movflags', '+faststart',
                output_file
            ], check=True, timeout=duration+20)
            
            if os.path.exists(output_file) and os.path.getsize(output_file) > 10000:
                print(f"Successfully recorded to {output_file} using curl+ffmpeg method")
            else:
                print("Recording with curl+ffmpeg failed or file is too small")
        
        except Exception as e:
            print(f"Error in curl+ffmpeg recording: {e}")
        
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            # Reset recording stats
            GDOTRequestHandler.recording_stats["active"] = False
    
    def _format_recording_name(self, filename):
        """Format a readable name from filename"""
        # Example: gdot_stream_20250412_123456.mp4 -> GDOT Stream 2025-04-12 12:34:56
        try:
            parts = filename.split('_')
            date_part = parts[-2]
            time_part = parts[-1].split('.')[0]
            
            formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]}"
            formatted_time = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:]}"
            
            return f"GDOT Stream {formatted_date} {formatted_time}"
        except:
            return filename

# Main function
def main():
    handler = GDOTRequestHandler
    
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Server started at http://localhost:{PORT}")
        print(f"GDOT Stream Viewer is now available")
        print(f"Recordings will be saved to: {os.path.abspath(RECORDINGS_DIR)}")
        print("Press Ctrl+C to stop the server")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        
        httpd.server_close()
        print("Server stopped")

if __name__ == "__main__":
    main()