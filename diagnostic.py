#!/usr/bin/env python3
"""
Diagnostic tool for GDOT Stream Viewer
Tests if the HLS stream is accessible and provides information about it
"""

import sys
import subprocess
import os
import re
import json
from datetime import datetime

def check_url(url):
    """Check if a URL is accessible using curl"""
    print(f"Checking URL: {url}")
    try:
        result = subprocess.run(
            ['curl', '-s', '-I', url],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if "200 OK" in result.stdout:
            print("✓ URL is accessible (200 OK)")
            return True
        else:
            print(f"✗ URL returned non-200 status")
            print(f"Response headers:\n{result.stdout}")
            return False
    except Exception as e:
        print(f"✗ Error checking URL: {e}")
        return False

def analyze_playlist(url):
    """Download and analyze the HLS playlist"""
    print(f"\nAnalyzing playlist: {url}")
    try:
        result = subprocess.run(
            ['curl', '-s', url],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0 or not result.stdout:
            print(f"✗ Failed to download playlist")
            return False
        
        print("✓ Playlist downloaded successfully")
        
        # Save playlist to file for inspection
        with open('diagnostic_playlist.m3u8', 'w') as f:
            f.write(result.stdout)
        print(f"✓ Saved playlist to diagnostic_playlist.m3u8")
        
        # Analyze playlist content
        playlist_content = result.stdout
        print("\nPlaylist content:")
        print("-" * 40)
        print(playlist_content[:500] + "..." if len(playlist_content) > 500 else playlist_content)
        print("-" * 40)
        
        # Extract chunklist URL
        chunklist_match = re.search(r'(chunklist.*\.m3u8)', playlist_content)
        if not chunklist_match:
            print("✗ No chunklist found in playlist")
            return False
        
        chunklist = chunklist_match.group(1)
        base_url = os.path.dirname(url)
        chunklist_url = f"{base_url}/{chunklist}"
        print(f"✓ Found chunklist: {chunklist_url}")
        
        # Check chunklist
        return analyze_chunklist(chunklist_url)
    except Exception as e:
        print(f"✗ Error analyzing playlist: {e}")
        return False

def analyze_chunklist(url):
    """Download and analyze the HLS chunklist"""
    print(f"\nAnalyzing chunklist: {url}")
    try:
        result = subprocess.run(
            ['curl', '-s', url],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0 or not result.stdout:
            print(f"✗ Failed to download chunklist")
            return False
        
        print("✓ Chunklist downloaded successfully")
        
        # Save chunklist to file for inspection
        with open('diagnostic_chunklist.m3u8', 'w') as f:
            f.write(result.stdout)
        print(f"✓ Saved chunklist to diagnostic_chunklist.m3u8")
        
        # Analyze chunklist content
        chunklist_content = result.stdout
        print("\nChunklist content:")
        print("-" * 40)
        print(chunklist_content[:500] + "..." if len(chunklist_content) > 500 else chunklist_content)
        print("-" * 40)
        
        # Extract segments
        segments = re.findall(r'media_\w+\.ts', chunklist_content)
        if not segments:
            print("✗ No media segments found in chunklist")
            return False
        
        print(f"✓ Found {len(segments)} media segments")
        
        # Check first segment
        segment = segments[0]
        base_url = os.path.dirname(url)
        segment_url = f"{base_url}/{segment}"
        
        print(f"\nChecking first segment: {segment_url}")
        segment_check = check_url(segment_url)
        
        return segment_check
    except Exception as e:
        print(f"✗ Error analyzing chunklist: {e}")
        return False

def run_ffmpeg_test(url):
    """Test if FFmpeg can process the stream"""
    print(f"\nTesting FFmpeg with stream: {url}")
    try:
        # Create a short 3-second test recording
        test_output = f"diagnostic_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        result = subprocess.run([
            'ffmpeg',
            '-y',                # Overwrite output files
            '-i', url,           # Input URL
            '-t', '3',           # 3 seconds duration
            '-c', 'copy',        # Copy streams without re-encoding
            '-v', 'warning',     # Less verbose output
            test_output
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode != 0 or not os.path.exists(test_output) or os.path.getsize(test_output) < 1000:
            print(f"✗ FFmpeg test failed")
            print(f"FFmpeg error: {result.stderr}")
            return False
        
        print(f"✓ FFmpeg test successful: created {test_output} ({os.path.getsize(test_output)} bytes)")
        return True
    except Exception as e:
        print(f"✗ Error in FFmpeg test: {e}")
        return False

def check_hls_js_compatibility():
    """Check if HLS.js version is compatible"""
    print("\nChecking HLS.js compatibility")
    with open('index.html', 'r') as f:
        content = f.read()
    
    hls_version_match = re.search(r'hls\.js/(\d+\.\d+\.\d+)', content)
    if not hls_version_match:
        print("✗ Could not determine HLS.js version")
        return
    
    version = hls_version_match.group(1)
    print(f"✓ Using HLS.js version {version}")
    
    # Specific known issues with different HLS.js versions
    compatibility_issues = {
        "0.5.": "Too old, lacks many features needed for GDOT streams",
        "0.8.": "Has issues with discontinuities in HLS streams",
        "0.12.": "Some streams with encryption may fail",
        "1.1.": "Has known issues with some chunklist formats",
    }
    
    for ver_prefix, issue in compatibility_issues.items():
        if version.startswith(ver_prefix):
            print(f"✗ Potential issue with HLS.js {version}: {issue}")
            print("  Consider upgrading to latest version (1.4.x+)")
            return
    
    if float(version.split('.')[0]) >= 1:
        print("✓ HLS.js version should be compatible with GDOT streams")
    else:
        print("⚠ HLS.js version is older than 1.0, consider upgrading")

def main():
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = "https://sfs-msc-pub-lq-01.navigator.dot.ga.gov:443/rtplive/ATL-CCTV-0092/playlist.m3u8"
        print(f"No URL provided, using default: {url}")
    
    print("\n===== GDOT Stream Diagnostic Tool =====")
    print(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n----- System Checks -----")
    # Check software versions
    try:
        ffmpeg_version = subprocess.check_output(['ffmpeg', '-version'], text=True).split('\n')[0]
        print(f"FFmpeg: {ffmpeg_version}")
    except:
        print("✗ FFmpeg not found or not working")
    
    try:
        curl_version = subprocess.check_output(['curl', '--version'], text=True).split('\n')[0]
        print(f"curl: {curl_version}")
    except:
        print("✗ curl not found or not working")
    
    # Check HLS.js compatibility
    check_hls_js_compatibility()
    
    print("\n----- Network Checks -----")
    # Basic URL check
    url_accessible = check_url(url)
    
    if url_accessible:
        # Analyze playlist structure
        playlist_valid = analyze_playlist(url)
        
        if playlist_valid:
            # Test FFmpeg
            ffmpeg_test = run_ffmpeg_test(url)
    
    print("\n===== Diagnostic Summary =====")
    print(f"URL: {url}")
    print(f"URL accessible: {'Yes' if url_accessible else 'No'}")
    
    if url_accessible:
        print(f"Playlist analysis: {'Successful' if playlist_valid else 'Failed'}")
        if playlist_valid:
            print(f"FFmpeg test: {'Successful' if ffmpeg_test else 'Failed'}")
    
    print("\nCheck diagnostic_playlist.m3u8 and diagnostic_chunklist.m3u8 for details")

if __name__ == "__main__":
    main()