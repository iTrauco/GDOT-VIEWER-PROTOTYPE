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
                if filename.endswith(('.mp4', '.ts', '.mkv')):  # Include all formats
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
                # Set content type based on file extension
                content_type = 'video/mp4'
                if filename.endswith('.ts'):
                    content_type = 'video/mp2t'
                elif filename.endswith('.mkv'):
                    content_type = 'video/x-matroska'
                
                self.send_response(200)
                self.send_header('Content-type', content_type)
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
        
        # Base name without extension for multiple formats
        base_output = os.path.join(RECORDINGS_DIR, f'gdot_stream_{timestamp}')
        mp4_output = f"{base_output}.mp4"
        
        # Set default output file for stats
        output_file = mp4_output
        
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

            # STRATEGY 1: Direct concatenation of TS segments
            # Create a file listing all segments for concat
            segments_list_file = os.path.join(temp_dir, 'segments.txt')
            with open(segments_list_file, 'w') as f:
                for i, segment in enumerate(segments[:segments_needed]):
                    local_segment = os.path.join(local_segments_dir, segment)
                    if os.path.exists(local_segment):
                        f.write(f"file '{local_segment}'\n")

            # First create a TS file by concatenating segments
            ts_output = f"{base_output}.ts"
            print("\nSTRATEGY 1: Creating TS file by direct concatenation...")
            ts_concat_cmd = [
                'ffmpeg',
                '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', segments_list_file,
                '-c', 'copy',
                ts_output
            ]
            print(" ".join(ts_concat_cmd))
            ts_success = False
            try:
                subprocess.run(ts_concat_cmd, check=True, timeout=duration+20, capture_output=True, text=True)
                if os.path.exists(ts_output) and os.path.getsize(ts_output) > 10000:
                    print(f"TS file created: {ts_output} ({os.path.getsize(ts_output)} bytes)")
                    ts_success = True
                else:
                    print(f"TS file creation failed or file is too small")
            except Exception as e:
                print(f"Error creating TS: {e}")

            # Convert TS to MP4 if TS creation was successful
            if ts_success:
                mp4_output = f"{base_output}_strategy1.mp4"
                print("Converting TS to MP4...")
                mp4_cmd = [
                    'ffmpeg',
                    '-y',
                    '-i', ts_output,
                    '-c:v', 'copy',
                    '-c:a', 'copy',
                    '-bsf:a', 'aac_adtstoasc',
                    '-movflags', '+faststart',
                    mp4_output
                ]
                print(" ".join(mp4_cmd))
                try:
                    subprocess.run(mp4_cmd, check=True, timeout=duration+20, capture_output=True, text=True)
                    if os.path.exists(mp4_output) and os.path.getsize(mp4_output) > 10000:
                        print(f"MP4 file created: {mp4_output} ({os.path.getsize(mp4_output)} bytes)")
                        # Check if moov atom is present
                        probe_cmd = ['ffprobe', '-v', 'error', mp4_output]
                        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                        if probe_result.returncode == 0:
                            print("MP4 verification successful - moov atom should be present")
                            # Update the reference output file if successful
                            output_file = mp4_output
                            GDOTRequestHandler.recording_stats["output_file"] = output_file
                        else:
                            print(f"MP4 verification failed: {probe_result.stderr}")
                    else:
                        print(f"MP4 creation failed or file is too small")
                except Exception as e:
                    print(f"Error creating MP4: {e}")

            # STRATEGY 2: Using HLS demuxer with direct output to MP4
            print("\nSTRATEGY 2: Using HLS demuxer with direct output to MP4...")
            mp4_output2 = f"{base_output}_strategy2.mp4"
            chunklist_file_modified = os.path.join(temp_dir, 'chunklist_modified.m3u8')
            
            # Create a modified chunklist with local paths
            with open(chunklist_file, 'r') as f_in, open(chunklist_file_modified, 'w') as f_out:
                content = f_in.read()
                # Modify paths to point to local segment files
                for segment in segments[:segments_needed]:
                    local_segment = os.path.join(local_segments_dir, segment)
                    content = content.replace(segment, local_segment)
                f_out.write(content)
                
            mp4_cmd2 = [
                'ffmpeg',
                '-y',
                '-allowed_extensions', 'ts',
                '-protocol_whitelist', 'file',
                '-i', chunklist_file_modified,
                '-c:v', 'copy',
                '-c:a', 'copy',
                '-bsf:a', 'aac_adtstoasc',
                '-movflags', '+faststart+frag_keyframe+empty_moov+default_base_moof',
                mp4_output2
            ]
            print(" ".join(mp4_cmd2))
            try:
                subprocess.run(mp4_cmd2, check=True, timeout=duration+20, capture_output=True, text=True)
                if os.path.exists(mp4_output2) and os.path.getsize(mp4_output2) > 10000:
                    print(f"MP4 file created (strategy 2): {mp4_output2} ({os.path.getsize(mp4_output2)} bytes)")
                    # Check if successful and update reference if needed
                    probe_cmd = ['ffprobe', '-v', 'error', mp4_output2]
                    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                    if probe_result.returncode == 0 and output_file != mp4_output2:
                        print("MP4 strategy 2 verification successful")
                        output_file = mp4_output2
                        GDOTRequestHandler.recording_stats["output_file"] = output_file
                else:
                    print(f"MP4 creation failed or file is too small (strategy 2)")
            except Exception as e:
                print(f"Error creating MP4 (strategy 2): {e}")

            # STRATEGY 3: Binary concatenation of TS segments
            print("\nSTRATEGY 3: Binary concatenation of TS segments...")
            ts_output3 = f"{base_output}_strategy3.ts"
            
            # Concatenate all segment files into one big TS file
            with open(ts_output3, 'wb') as outfile:
                for i, segment in enumerate(segments[:segments_needed]):
                    local_segment = os.path.join(local_segments_dir, segment)
                    if os.path.exists(local_segment):
                        with open(local_segment, 'rb') as infile:
                            outfile.write(infile.read())
            
            if os.path.exists(ts_output3) and os.path.getsize(ts_output3) > 10000:
                print(f"TS file created (binary concat): {ts_output3} ({os.path.getsize(ts_output3)} bytes)")
                
                # Convert to MP4
                mp4_output3 = f"{base_output}_strategy3.mp4"
                mp4_cmd3 = [
                    'ffmpeg',
                    '-y',
                    '-i', ts_output3,
                    '-c:v', 'copy',
                    '-c:a', 'copy',
                    '-bsf:a', 'aac_adtstoasc',
                    '-movflags', '+faststart',
                    mp4_output3
                ]
                print(" ".join(mp4_cmd3))
                try:
                    subprocess.run(mp4_cmd3, check=True, timeout=duration+20, capture_output=True, text=True)
                    if os.path.exists(mp4_output3) and os.path.getsize(mp4_output3) > 10000:
                        print(f"MP4 file created (strategy 3): {mp4_output3} ({os.path.getsize(mp4_output3)} bytes)")
                        # Check if successful and update reference if needed
                        probe_cmd = ['ffprobe', '-v', 'error', mp4_output3]
                        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                        if probe_result.returncode == 0 and output_file != mp4_output3:
                            print("MP4 strategy 3 verification successful")
                            output_file = mp4_output3
                            GDOTRequestHandler.recording_stats["output_file"] = output_file
                    else:
                        print(f"MP4 creation failed or file is too small (strategy 3)")
                except Exception as e:
                    print(f"Error creating MP4 (strategy 3): {e}")
            else:
                print(f"TS file creation failed (binary concat)")

            # Update stats with final output file info
            if os.path.exists(output_file):
                size_bytes = os.path.getsize(output_file)
                if size_bytes < 1024:
                    GDOTRequestHandler.recording_stats["file_size"] = f"{size_bytes} B"
                elif size_bytes < 1024 * 1024:
                    GDOTRequestHandler.recording_stats["file_size"] = f"{size_bytes/1024:.1f} KB"
                else:
                    GDOTRequestHandler.recording_stats["file_size"] = f"{size_bytes/(1024*1024):.1f} MB"
                    
                print(f"\nFinal output file: {output_file} ({GDOTRequestHandler.recording_stats['file_size']})")
            else:
                print("\nNo valid output file was created.")
                
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
            # Split by dots and take the part before the extension
            name_part = filename.split('.')[0]
            
            # Handle strategy suffixes
            if '_strategy' in name_part:
                name_part = name_part.split('_strategy')[0]
                
            parts = name_part.split('_')
            date_part = parts[-2]
            time_part = parts[-1]
            
            formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]}"
            formatted_time = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:]}"
            
            # Add format to the name
            format_ext = filename.split('.')[-1].upper()
            
            # Add strategy if present
            strategy = ""
            if '_strategy' in filename:
                strategy_num = filename.split('_strategy')[1][0]
                strategy = f" (Strategy {strategy_num})"
            
            return f"GDOT Stream {formatted_date} {formatted_time} {format_ext}{strategy}"
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