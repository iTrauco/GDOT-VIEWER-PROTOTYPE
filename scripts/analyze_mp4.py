#!/usr/bin/env python3
"""
MP4 file analyzer to diagnose playback issues with GDOT recordings
"""

import os
import sys
import subprocess
import json
import binascii
import struct

def analyze_file(filepath):
    """Analyze an MP4 file using multiple methods"""
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist")
        return
    
    filesize = os.path.getsize(filepath)
    print(f"\n=== Analyzing {filepath} ({filesize} bytes) ===\n")
    
    # Basic file header check
    print("--- File Header Analysis ---")
    with open(filepath, 'rb') as f:
        header = f.read(32)  # Read first 32 bytes
        
        print(f"First 32 bytes (hex): {binascii.hexlify(header).decode()}")
        
        # Check for valid MP4 file header (ftyp box)
        if b'ftyp' in header:
            ftyp_pos = header.find(b'ftyp')
            print(f"Found 'ftyp' at position {ftyp_pos}")
            
            # Try to extract major brand
            if ftyp_pos >= 4 and ftyp_pos + 8 <= len(header):
                brand = header[ftyp_pos+4:ftyp_pos+8].decode('utf-8', errors='replace')
                print(f"Major brand: {brand}")
            
            # Look for 'mdat' or 'moov' boxes
            with open(filepath, 'rb') as full_file:
                content = full_file.read(min(filesize, 10000))  # Read up to 10KB
                if b'mdat' in content:
                    print("Found 'mdat' box (media data)")
                else:
                    print("WARNING: No 'mdat' box found in first 10KB")
                
                if b'moov' in content:
                    print("Found 'moov' box (movie metadata)")
                else:
                    print("WARNING: No 'moov' box found in first 10KB")
        else:
            print("WARNING: No 'ftyp' signature found - this may not be a valid MP4 file")
    
    # Check beginning/end of file
    print("\n--- Beginning/End of File ---")
    try:
        output = subprocess.check_output(f"hexdump -C {filepath} | head -10", shell=True, 
                                        stderr=subprocess.STDOUT, text=True)
        print("Beginning of file:\n" + output)
        
        output = subprocess.check_output(f"hexdump -C {filepath} | tail -5", shell=True, 
                                        stderr=subprocess.STDOUT, text=True)
        print("End of file:\n" + output)
    except subprocess.CalledProcessError as e:
        print(f"Error running hexdump: {e}")
    
    # Run FFprobe analysis
    print("\n--- FFprobe Analysis ---")
    try:
        ffprobe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_format',
            '-show_streams',
            '-print_format', 'json',
            filepath
        ]
        result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout:
            try:
                probe_data = json.loads(result.stdout)
                
                # Format information
                if 'format' in probe_data:
                    format_info = probe_data['format']
                    print(f"Format: {format_info.get('format_name', 'Unknown')}")
                    print(f"Duration: {format_info.get('duration', 'Unknown')} seconds")
                    print(f"Size: {format_info.get('size', 'Unknown')} bytes")
                    print(f"Bitrate: {format_info.get('bit_rate', 'Unknown')} bps")
                
                # Stream information
                if 'streams' in probe_data:
                    streams = probe_data['streams']
                    print(f"\nFound {len(streams)} stream(s):")
                    
                    for i, stream in enumerate(streams):
                        stream_type = stream.get('codec_type', 'Unknown')
                        codec = stream.get('codec_name', 'Unknown')
                        print(f"  Stream #{i}: {stream_type}, Codec: {codec}")
                        
                        if stream_type == 'video':
                            print(f"    Resolution: {stream.get('width', '?')}x{stream.get('height', '?')}")
                            print(f"    Frame rate: {stream.get('r_frame_rate', 'Unknown')}")
                        elif stream_type == 'audio':
                            print(f"    Sample rate: {stream.get('sample_rate', 'Unknown')} Hz")
                            print(f"    Channels: {stream.get('channels', 'Unknown')}")
            except json.JSONDecodeError:
                print("Error: Could not parse FFprobe output")
                print(f"Raw FFprobe output: {result.stdout}")
        else:
            print(f"FFprobe error: {result.stderr}")
            
    except FileNotFoundError:
        print("FFprobe not found. Please install FFmpeg/FFprobe.")
    
    # Run MediaInfo if available
    print("\n--- MediaInfo Analysis ---")
    try:
        mediainfo_cmd = ['mediainfo', '--Output=JSON', filepath]
        result = subprocess.run(mediainfo_cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout:
            try:
                mediainfo_data = json.loads(result.stdout)
                print("MediaInfo analysis successful")
                
                # Extract key details
                if 'media' in mediainfo_data and 'track' in mediainfo_data['media']:
                    tracks = mediainfo_data['media']['track']
                    for track in tracks:
                        if track.get('@type') == 'General':
                            print(f"Format: {track.get('Format', 'Unknown')}")
                            print(f"Complete name: {track.get('CompleteName', 'Unknown')}")
                            print(f"Format profile: {track.get('Format_Profile', 'Unknown')}")
                            print(f"File size: {track.get('FileSize', 'Unknown')}")
            except json.JSONDecodeError:
                print("Error: Could not parse MediaInfo output")
                print(f"Raw MediaInfo output: {result.stdout}")
        else:
            print("MediaInfo error or not available")
            
    except FileNotFoundError:
        print("MediaInfo not found. FFprobe results should be sufficient.")

def analyze_directory(directory, extension='.mp4'):
    """Analyze all MP4 files in a directory"""
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        return
    
    mp4_files = [f for f in os.listdir(directory) if f.endswith(extension)]
    
    if not mp4_files:
        print(f"No {extension} files found in {directory}")
        return
    
    print(f"Found {len(mp4_files)} {extension} files to analyze")
    
    for mp4_file in mp4_files:
        filepath = os.path.join(directory, mp4_file)
        analyze_file(filepath)

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_mp4.py <file_or_directory> [extension]")
        print("Example: python analyze_mp4.py recordings/")
        print("Example: python analyze_mp4.py recordings/gdot_stream_20250411.mp4")
        print("Example: python analyze_mp4.py recordings/ .ts")
        return
    
    path = sys.argv[1]
    extension = sys.argv[2] if len(sys.argv) > 2 else '.mp4'
    
    if os.path.isdir(path):
        analyze_directory(path, extension)
    elif os.path.isfile(path):
        analyze_file(path)
    else:
        print(f"Error: {path} is not a valid file or directory")

if __name__ == "__main__":
    main()