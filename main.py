import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import zipfile
from datetime import datetime
import shutil
import io
import uuid
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import speech_recognition as sr
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max
app.config['UPLOAD_FOLDER'] = '/tmp/audio_uploads'
app.config['OUTPUT_FOLDER'] = '/tmp/audio_outputs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'ogg', 'm4a', 'aac', 'wma'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_audio_metadata(audio_path):
    """Extract comprehensive metadata from audio file"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # File metadata
        file_stats = os.stat(audio_path)
        file_size = file_stats.st_size
        
        # Detect format
        file_ext = audio_path.rsplit('.', 1)[1].lower()
        
        metadata = {
            'filename': os.path.basename(audio_path),
            'file_size_bytes': file_size,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'format': file_ext.upper(),
            'duration_seconds': round(duration, 2),
            'duration_formatted': f"{int(duration // 60)}:{int(duration % 60):02d}",
            'sample_rate': sr,
            'channels': 1 if len(y.shape) == 1 else y.shape[0],
            'total_samples': len(y),
            'bitrate_estimate': round((file_size * 8) / duration / 1000, 2) if duration > 0 else 0,
            'created': datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
        }
        
        # Generate hashes
        with open(audio_path, 'rb') as f:
            file_bytes = f.read()
            metadata['md5'] = hashlib.md5(file_bytes).hexdigest()
            metadata['sha256'] = hashlib.sha256(file_bytes).hexdigest()
        
        return metadata
    
    except Exception as e:
        return {'error': f'Metadata extraction failed: {str(e)}'}


def spectral_analysis(audio_path, output_dir):
    """Analyze frequency spectrum and detect anomalies"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        # Generate spectrogram
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Plot spectrogram
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram Analysis')
        plt.tight_layout()
        spectrogram_path = os.path.join(output_dir, 'spectrogram.png')
        plt.savefig(spectrogram_path)
        plt.close()
        
        # Detect spectral anomalies
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_flux = np.sqrt(np.mean(np.diff(np.abs(D), axis=1)**2, axis=0))
        
        # Calculate anomaly score
        centroid_std = np.std(spectral_centroid)
        flux_mean = np.mean(spectral_flux)
        
        anomaly_score = (centroid_std / 1000) + (flux_mean * 10)
        
        return {
            'spectrogram': spectrogram_path,
            'spectral_centroid_mean': round(float(np.mean(spectral_centroid)), 2),
            'spectral_centroid_std': round(float(centroid_std), 2),
            'spectral_rolloff_mean': round(float(np.mean(spectral_rolloff)), 2),
            'anomaly_score': round(float(anomaly_score), 2),
            'interpretation': 'High' if anomaly_score > 10 else 'Medium' if anomaly_score > 5 else 'Low'
        }
    
    except Exception as e:
        return {'error': f'Spectral analysis failed: {str(e)}'}


def waveform_analysis(audio_path, output_dir):
    """Analyze waveform for splice detection and amplitude patterns"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        # Plot waveform
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(y, sr=sr)
        plt.title('Waveform Analysis')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        waveform_path = os.path.join(output_dir, 'waveform.png')
        plt.savefig(waveform_path)
        plt.close()
        
        # Detect sudden amplitude changes (potential splices)
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Calculate amplitude differences
        rms_diff = np.abs(np.diff(rms))
        threshold = np.mean(rms_diff) + 2 * np.std(rms_diff)
        splice_candidates = np.where(rms_diff > threshold)[0]
        
        splice_score = len(splice_candidates) / len(rms_diff) * 100
        
        return {
            'waveform': waveform_path,
            'rms_mean': round(float(np.mean(rms)), 4),
            'rms_std': round(float(np.std(rms)), 4),
            'splice_candidates': int(len(splice_candidates)),
            'splice_score': round(float(splice_score), 2),
            'interpretation': 'High' if splice_score > 5 else 'Medium' if splice_score > 2 else 'Low'
        }
    
    except Exception as e:
        return {'error': f'Waveform analysis failed: {str(e)}'}


def noise_analysis(audio_path, output_dir):
    """Analyze background noise consistency"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        # Detect non-speech segments (likely background noise)
        intervals = librosa.effects.split(y, top_db=20)
        
        # Analyze noise in silent segments
        noise_segments = []
        for i in range(len(intervals) - 1):
            start = intervals[i][1]
            end = intervals[i + 1][0]
            if end - start > sr * 0.1:  # At least 0.1s
                noise_segment = y[start:end]
                noise_segments.append(np.std(noise_segment))
        
        if len(noise_segments) > 1:
            noise_consistency = np.std(noise_segments)
            inconsistency_score = noise_consistency * 1000
        else:
            inconsistency_score = 0
        
        # Zero-crossing rate analysis
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_std = np.std(zcr)
        
        total_score = inconsistency_score + (zcr_std * 100)
        
        return {
            'noise_segments_analyzed': len(noise_segments),
            'noise_inconsistency': round(float(inconsistency_score), 2),
            'zero_crossing_rate_std': round(float(zcr_std), 4),
            'total_noise_score': round(float(total_score), 2),
            'interpretation': 'High' if total_score > 15 else 'Medium' if total_score > 8 else 'Low'
        }
    
    except Exception as e:
        return {'error': f'Noise analysis failed: {str(e)}'}


def tampering_detection(audio_path, output_dir):
    """Detect signs of tampering: cuts, splices, re-encoding"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        # Detect abrupt transitions
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        
        # Calculate onset density
        duration = librosa.get_duration(y=y, sr=sr)
        onset_density = len(onset_frames) / duration
        
        # Spectral contrast (detects unnatural frequency patterns)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_std = np.std(contrast)
        
        # Phase discontinuity detection
        phase = np.angle(librosa.stft(y))
        phase_diff = np.diff(phase, axis=1)
        phase_discontinuities = np.sum(np.abs(phase_diff) > np.pi * 0.9)
        discontinuity_rate = phase_discontinuities / phase.shape[1]
        
        # Calculate tampering likelihood
        tampering_score = 0
        if onset_density > 10:
            tampering_score += 30
        elif onset_density > 5:
            tampering_score += 15
        
        if contrast_std > 15:
            tampering_score += 25
        elif contrast_std > 10:
            tampering_score += 15
        
        if discontinuity_rate > 0.1:
            tampering_score += 30
        elif discontinuity_rate > 0.05:
            tampering_score += 15
        
        return {
            'onset_density': round(float(onset_density), 2),
            'spectral_contrast_std': round(float(contrast_std), 2),
            'phase_discontinuity_rate': round(float(discontinuity_rate), 4),
            'tampering_score': min(tampering_score, 100),
            'interpretation': 'High' if tampering_score > 50 else 'Medium' if tampering_score > 25 else 'Low'
        }
    
    except Exception as e:
        return {'error': f'Tampering detection failed: {str(e)}'}


def voice_activity_detection(audio_path, output_dir):
    """Detect speech segments, count speakers, and transcribe"""
    try:
        y, sr_rate = librosa.load(audio_path, sr=None)
        
        # Voice activity detection
        intervals = librosa.effects.split(y, top_db=20)
        
        # Calculate speech/silence ratio
        speech_duration = sum([end - start for start, end in intervals]) / sr_rate
        total_duration = len(y) / sr_rate
        silence_duration = total_duration - speech_duration
        speech_ratio = (speech_duration / total_duration) * 100
        
        # Plot VAD
        plt.figure(figsize=(12, 4))
        times = librosa.times_like(y, sr=sr_rate)
        plt.plot(times, y, alpha=0.5, label='Audio')
        for start, end in intervals:
            plt.axvspan(start/sr_rate, end/sr_rate, alpha=0.3, color='green', 
                       label='Speech' if start == intervals[0][0] else '')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Voice Activity Detection')
        plt.legend()
        plt.tight_layout()
        vad_path = os.path.join(output_dir, 'voice_activity.png')
        plt.savefig(vad_path)
        plt.close()
        
        # SPEAKER COUNT
        speaker_count = 1
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=13)
            speech_mfcc = []
            for start, end in intervals:
                start_frame = librosa.time_to_frames(start/sr_rate, sr=sr_rate)
                end_frame = librosa.time_to_frames(end/sr_rate, sr=sr_rate)
                if end_frame > start_frame:
                    speech_mfcc.append(mfcc[:, start_frame:end_frame].T)
            
            if speech_mfcc:
                all_mfcc = np.vstack(speech_mfcc)
                best_k = 1
                if len(all_mfcc) > 100:
                    for k in range(1, min(6, len(all_mfcc)//50)):
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        kmeans.fit(all_mfcc)
                        if kmeans.inertia_ < 1000:
                            best_k = k
                speaker_count = best_k
        except:
            speaker_count = 1
        
        # TRANSCRIPTION - IMPROVED WITH CHUNKING
        transcription = ""
        try:
            # Limit to first 90 seconds
            max_duration = min(90, total_duration)
            if sr_rate != 16000:
                y_16k = librosa.resample(y[:int(max_duration*sr_rate)], orig_sr=sr_rate, target_sr=16000)
            else:
                y_16k = y[:int(max_duration*16000)]
            
            temp_wav = os.path.join(output_dir, 'temp_transcribe.wav')
            sf.write(temp_wav, y_16k, 16000)
            
            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = False
            
            with sr.AudioFile(temp_wav) as source:
                # Process in 30-second chunks
                chunk_duration = 30
                full_text = []
                
                for i in range(0, int(max_duration), chunk_duration):
                    try:
                        audio_chunk = recognizer.record(source, duration=min(chunk_duration, max_duration - i))
                        text = recognizer.recognize_google(audio_chunk, language='en-US')
                        full_text.append(text)
                    except sr.UnknownValueError:
                        continue
                    except sr.RequestError:
                        break
                    except:
                        break
                
                transcription = ' '.join(full_text) if full_text else "No clear speech detected"
            
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
                
        except Exception as e:
            transcription = "Transcription unavailable (audio may be unclear or service offline)"
        
        return {
            'vad_plot': vad_path,
            'total_duration': round(total_duration, 2),
            'speech_duration': round(speech_duration, 2),
            'silence_duration': round(silence_duration, 2),
            'speech_ratio': round(speech_ratio, 2),
            'segments_detected': len(intervals),
            'speaker_count': speaker_count,
            'transcription': transcription
        }
    
    except Exception as e:
        return {
            'vad_plot': None,
            'speaker_count': 1,
            'transcription': 'Analysis error',
            'error': str(e)[:100]
        }


def advanced_noise_floor_analysis(audio_path, output_dir):
    """Detect noise floor inconsistencies across segments"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        # Divide audio into 1-second windows
        window_size = sr  # 1 second
        num_windows = len(y) // window_size
        
        noise_floors = []
        timestamps = []
        
        for i in range(num_windows):
            segment = y[i*window_size:(i+1)*window_size]
            # Get noise floor (bottom 5% of magnitude spectrum)
            D = np.abs(librosa.stft(segment))
            noise_floor = np.percentile(D, 5)
            noise_floors.append(noise_floor)
            timestamps.append(i)
        
        # Detect sudden changes
        noise_diffs = np.abs(np.diff(noise_floors))
        threshold = np.mean(noise_diffs) + 2 * np.std(noise_diffs)
        anomaly_points = np.where(noise_diffs > threshold)[0]
        
        # Plot
        plt.figure(figsize=(12, 4))
        plt.plot(timestamps, noise_floors, label='Noise Floor')
        plt.scatter(anomaly_points, [noise_floors[i] for i in anomaly_points], 
                   color='red', s=100, label='Anomalies', zorder=5)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Noise Floor Magnitude')
        plt.title('Noise Floor Consistency Analysis')
        plt.legend()
        plt.tight_layout()
        noise_path = os.path.join(output_dir, 'noise_floor_analysis.png')
        plt.savefig(noise_path)
        plt.close()
        
        return {
            'noise_floor_plot': noise_path,
            'anomaly_count': int(len(anomaly_points)),
            'anomaly_timestamps': [int(t) for t in anomaly_points],
            'noise_floor_std': round(float(np.std(noise_floors)), 6),
            'interpretation': 'High' if len(anomaly_points) > 5 else 'Medium' if len(anomaly_points) > 2 else 'Low'
        }
    except Exception as e:
        return {'error': f'Noise floor analysis failed: {str(e)}'}


def harmonic_continuity_analysis(audio_path):
    """Detect unnatural gaps in harmonic structure (AI voice detection)"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extract harmonic component
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Chromagram (pitch class profiles)
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        
        # Detect harmonic gaps (missing fundamental frequencies)
        harmonic_energy = np.sum(chroma, axis=0)
        gaps = harmonic_energy < (np.mean(harmonic_energy) * 0.3)
        gap_percentage = (np.sum(gaps) / len(gaps)) * 100
        
        # F0 (pitch) tracking
        f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        
        # Detect unnatural pitch jumps (>1 octave)
        pitch_jumps = np.abs(np.diff(f0[f0 > 0]))
        unnatural_jumps = np.sum(pitch_jumps > 200)  # Hz
        
        ai_likelihood = 0
        if gap_percentage > 15:
            ai_likelihood += 40
        if unnatural_jumps > 10:
            ai_likelihood += 30
        if np.std(f0[f0 > 0]) < 10:  # Too stable = synthetic
            ai_likelihood += 30
        
        return {
            'harmonic_gap_percentage': round(float(gap_percentage), 2),
            'unnatural_pitch_jumps': int(unnatural_jumps),
            'pitch_stability': round(float(np.std(f0[f0 > 0])), 2),
            'ai_voice_likelihood': min(ai_likelihood, 100),
            'interpretation': 'High AI Likelihood' if ai_likelihood > 60 else 'Possible AI' if ai_likelihood > 30 else 'Likely Human'
        }
    except Exception as e:
        return {'error': f'Harmonic analysis failed: {str(e)}'}


def breath_sound_detection(audio_path):
    """Detect missing breath sounds (sign of editing)"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        # Detect low-energy regions (potential breath locations)
        rms = librosa.feature.rms(y=y)[0]
        low_energy = rms < (np.mean(rms) * 0.3)
        
        # Breaths are typically 0.2-0.5 seconds
        hop_length = 512
        frame_duration = hop_length / sr
        
        breath_candidates = []
        in_breath = False
        breath_start = 0
        
        for i, is_low in enumerate(low_energy):
            if is_low and not in_breath:
                breath_start = i
                in_breath = True
            elif not is_low and in_breath:
                breath_duration = (i - breath_start) * frame_duration
                if 0.1 < breath_duration < 0.8:  # Typical breath duration
                    breath_candidates.append(breath_start * frame_duration)
                in_breath = False
        
        # Expected breath rate: ~12-20 per minute for speech
        duration_minutes = len(y) / sr / 60
        expected_breaths = duration_minutes * 15
        detected_breaths = len(breath_candidates)
        
        breath_deficit = max(0, expected_breaths - detected_breaths)
        missing_percentage = (breath_deficit / expected_breaths * 100) if expected_breaths > 0 else 0
        
        return {
            'expected_breaths': round(expected_breaths, 1),
            'detected_breaths': detected_breaths,
            'breath_deficit': round(breath_deficit, 1),
            'missing_percentage': round(float(missing_percentage), 2),
            'breath_timestamps': [round(t, 2) for t in breath_candidates[:20]],  # First 20
            'interpretation': 'High Editing' if missing_percentage > 50 else 'Moderate' if missing_percentage > 25 else 'Natural'
        }
    except Exception as e:
        return {'error': f'Breath detection failed: {str(e)}'}


def compression_history_analysis(audio_path):
    """Detect multiple re-encoding cycles (sign of editing)"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        # Analyze quantization artifacts
        D = librosa.stft(y)
        magnitude = np.abs(D)
        
        # Double compression leaves specific patterns in histogram
        hist, bins = np.histogram(magnitude.flatten(), bins=100)
        
        # Look for double peaks (sign of re-quantization)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(hist, distance=5)
        
        # Check for blockiness in spectrogram (compression artifact)
        block_variance = []
        block_size = 8  # Typical DCT block size
        for i in range(0, magnitude.shape[1] - block_size, block_size):
            block = magnitude[:, i:i+block_size]
            block_variance.append(np.var(block))
        
        block_regularity = np.std(block_variance)
        
        # Multiple encoding = low variance between blocks
        reencoding_score = 0
        if len(peaks) > 3:
            reencoding_score += 40
        if block_regularity < np.mean(block_variance) * 0.5:
            reencoding_score += 40
        
        # Check for clipped samples (lossy compression indicator)
        clipping_threshold = 0.99
        clipped_samples = np.sum(np.abs(y) > clipping_threshold)
        if clipped_samples > len(y) * 0.01:  # More than 1%
            reencoding_score += 20
        
        return {
            'histogram_peaks': int(len(peaks)),
            'block_regularity_score': round(float(block_regularity), 4),
            'clipped_samples': int(clipped_samples),
            'reencoding_likelihood': min(reencoding_score, 100),
            'interpretation': 'Multiple Encodings Detected' if reencoding_score > 60 else 'Possible Re-encoding' if reencoding_score > 30 else 'Single Encoding'
        }
    except Exception as e:
        return {'error': f'Compression analysis failed: {str(e)}'}


def copy_move_detection(audio_path, output_dir):
    """Detect duplicated audio segments (fake backgrounds, looped noise)"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        # Use MFCC fingerprints to find similar segments
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Compare all segments with each other
        segment_length = 50  # frames
        duplicates = []
        
        for i in range(0, mfcc.shape[1] - segment_length, 10):
            segment1 = mfcc[:, i:i+segment_length]
            
            for j in range(i + segment_length*2, mfcc.shape[1] - segment_length, 10):
                segment2 = mfcc[:, j:j+segment_length]
                
                # Cosine similarity
                similarity = np.dot(segment1.flatten(), segment2.flatten()) / (
                    np.linalg.norm(segment1.flatten()) * np.linalg.norm(segment2.flatten())
                )
                
                if similarity > 0.95:  # Very similar
                    time1 = librosa.frames_to_time(i, sr=sr)
                    time2 = librosa.frames_to_time(j, sr=sr)
                    duplicates.append((round(time1, 2), round(time2, 2), round(float(similarity), 3)))
        
        # Plot duplicates
        if duplicates:
            plt.figure(figsize=(12, 6))
            for dup in duplicates[:20]:  # Plot first 20
                plt.scatter([dup[0]], [dup[1]], s=100, alpha=0.6)
            plt.xlabel('First Occurrence (seconds)')
            plt.ylabel('Duplicate Occurrence (seconds)')
            plt.title('Copy-Move Detection')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            copymove_path = os.path.join(output_dir, 'copy_move.png')
            plt.savefig(copymove_path)
            plt.close()
        else:
            copymove_path = None
        
        return {
            'copy_move_plot': copymove_path,
            'duplicate_segments_found': len(duplicates),
            'duplicate_pairs': duplicates[:10],  # First 10
            'interpretation': 'High Duplication' if len(duplicates) > 10 else 'Some Duplication' if len(duplicates) > 3 else 'No Significant Duplication'
        }
    except Exception as e:
        return {'error': f'Copy-move detection failed: {str(e)}'}


def precise_splice_detection(audio_path, output_dir):
    """Find exact splice points with timestamps"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        # Multi-method splice detection
        splice_points = []
        
        # Method 1: Phase discontinuity
        D = librosa.stft(y)
        phase = np.angle(D)
        phase_diff = np.diff(np.unwrap(phase, axis=1), axis=1)
        phase_variance = np.var(phase_diff, axis=0)
        phase_threshold = np.mean(phase_variance) + 3 * np.std(phase_variance)
        phase_splices = np.where(phase_variance > phase_threshold)[0]
        
        # Method 2: Spectral flux discontinuity
        flux = np.sqrt(np.sum(np.diff(np.abs(D), axis=1)**2, axis=0))
        flux_threshold = np.mean(flux) + 3 * np.std(flux)
        flux_splices = np.where(flux > flux_threshold)[0]
        
        # Method 3: Amplitude discontinuity
        rms = librosa.feature.rms(y=y)[0]
        rms_diff = np.abs(np.diff(rms))
        rms_threshold = np.mean(rms_diff) + 3 * np.std(rms_diff)
        rms_splices = np.where(rms_diff > rms_threshold)[0]
        
        # Combine all methods (consensus approach)
        all_splices = np.concatenate([phase_splices, flux_splices, rms_splices])
        unique_splices, counts = np.unique(all_splices, return_counts=True)
        
        # Keep only splices detected by 2+ methods
        confident_splices = unique_splices[counts >= 2]
        
        # Convert to timestamps
        for frame in confident_splices:
            timestamp = librosa.frames_to_time(frame, sr=sr)
            splice_points.append(round(timestamp, 3))
        
        # Plot splice points on waveform
        plt.figure(figsize=(14, 5))
        times = librosa.times_like(y, sr=sr)
        plt.plot(times, y, alpha=0.7, linewidth=0.5)
        for sp in splice_points:
            plt.axvline(x=sp, color='red', linestyle='--', linewidth=2, alpha=0.7)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.title(f'Splice Detection - {len(splice_points)} splice points found')
        plt.tight_layout()
        splice_path = os.path.join(output_dir, 'splice_detection.png')
        plt.savefig(splice_path)
        plt.close()
        
        return {
            'splice_detection_plot': splice_path,
            'splice_count': len(splice_points),
            'splice_timestamps': splice_points,
            'confidence': 'High' if len(confident_splices) > 5 else 'Medium' if len(confident_splices) > 2 else 'Low'
        }
    except Exception as e:
        return {'error': f'Splice detection failed: {str(e)}'}


def audio_enhancement(audio_path, output_dir):
    """Apply noise reduction and normalization"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        # Noise reduction (simple spectral gating)
        D = librosa.stft(y)
        magnitude = np.abs(D)
        phase = np.angle(D)
        
        # Estimate noise floor
        noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)
        mask = magnitude > (noise_floor * 2)
        magnitude_cleaned = magnitude * mask
        
        # Reconstruct
        D_cleaned = magnitude_cleaned * np.exp(1j * phase)
        y_cleaned = librosa.istft(D_cleaned)
        
        # Normalize
        y_normalized = librosa.util.normalize(y_cleaned)
        
        # Save enhanced audio
        enhanced_path = os.path.join(output_dir, 'enhanced_audio.wav')
        sf.write(enhanced_path, y_normalized, sr)
        
        return {
            'enhanced_audio': enhanced_path,
            'noise_reduction_applied': True,
            'normalization_applied': True
        }
    
    except Exception as e:
        return {'error': f'Enhancement failed: {str(e)}'}


def compare_audio_files(audio1_path, audio2_path):
    """Compare two audio files for similarity"""
    try:
        y1, sr1 = librosa.load(audio1_path, sr=None)
        y2, sr2 = librosa.load(audio2_path, sr=None)
        
        # Resample if necessary
        if sr1 != sr2:
            y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)
            sr2 = sr1
        
        # Match lengths
        min_len = min(len(y1), len(y2))
        y1 = y1[:min_len]
        y2 = y2[:min_len]
        
        # Calculate similarity using cross-correlation
        correlation = np.correlate(y1, y2, mode='valid')[0]
        max_correlation = np.sqrt(np.sum(y1**2) * np.sum(y2**2))
        similarity = (correlation / max_correlation) if max_correlation > 0 else 0
        
        # Spectral similarity
        mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)
        mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13)
        
        min_frames = min(mfcc1.shape[1], mfcc2.shape[1])
        spectral_similarity = np.mean(np.abs(mfcc1[:, :min_frames] - mfcc2[:, :min_frames]))
        
        return {
            'waveform_similarity': round(float(np.abs(similarity)), 4),
            'spectral_distance': round(float(spectral_similarity), 4),
            'interpretation': 'Very Similar' if similarity > 0.8 else 'Similar' if similarity > 0.5 else 'Different'
        }
    
    except Exception as e:
        return {'error': f'Comparison failed: {str(e)}'}


def generate_user_friendly_summary(report):
    """Generate simplified, non-technical summary"""
    try:
        # Get scores
        tampering_score = report['tampering_detection'].get('tampering_score', 0)
        spectral_anomaly = report['spectral_analysis'].get('anomaly_score', 0)
        splice_score = report['waveform_analysis'].get('splice_score', 0)
        noise_score = report['noise_analysis'].get('total_noise_score', 0)
        ai_likelihood = report.get('harmonic_analysis', {}).get('ai_voice_likelihood', 0)
        breath_missing = report.get('breath_analysis', {}).get('missing_percentage', 0)
        reencoding = report.get('compression_analysis', {}).get('reencoding_likelihood', 0)
        duplicates = report.get('copy_move_analysis', {}).get('duplicate_segments_found', 0)
        splice_count = report.get('splice_detection', {}).get('splice_count', 0)
        
        # Overall score
        overall_score = (
            tampering_score * 0.2 +
            min(spectral_anomaly * 5, 30) * 0.15 +
            min(splice_score * 10, 30) * 0.15 +
            ai_likelihood * 0.2 +
            breath_missing * 0.1 +
            reencoding * 0.1 +
            min(duplicates * 2, 20) * 0.05 +
            min(splice_count * 3, 15) * 0.05
        )
        
        if overall_score > 60:
            status = "🔴 HIGH RISK"
            recommendation = "This audio shows strong signs of manipulation. Do not trust without verification."
        elif overall_score > 30:
            status = "🟡 MEDIUM RISK"
            recommendation = "This audio shows suspicious patterns. Verify with other sources."
        else:
            status = "🟢 LOW RISK"
            recommendation = "This audio appears mostly authentic, but always verify important claims."
        
        # SIMPLIFIED FINDINGS
        spectral_finding = "Normal frequency patterns" if spectral_anomaly <= 5 else \
                          "Unusual frequency patterns detected (possible editing)"
        
        vad_finding = f"Natural speech rhythm with {report['vad'].get('speaker_count', 1)} speaker(s) detected"
        if breath_missing > 50:
            vad_finding += " - Missing breath sounds indicate heavy editing"
        
        tampering_finding = "No significant tampering detected" if tampering_score <= 25 else \
                           f"Audio has been edited - {splice_count} splice points found"
        
        metadata = report['metadata']
        
        return {
            'status': status,
            'authenticity_score': round(100 - overall_score, 2),
            'tampering_probability': f"{round(overall_score, 2)}%",
            
            # SIMPLIFIED ANALYSIS
            'spectral_analysis_summary': spectral_finding,
            'voice_activity_summary': vad_finding,
            'tampering_summary': tampering_finding,
            'speaker_count': report['vad'].get('speaker_count', 1),
            'transcription': report['vad'].get('transcription', 'N/A'),
            
            'recommendation': recommendation,
            'splice_locations': report.get('splice_detection', {}).get('splice_timestamps', []),
            
            # BASIC INFO
            'audio_info': {
                'format': metadata.get('format', 'Unknown'),
                'duration': metadata.get('duration_formatted', 'Unknown'),
                'sample_rate': f"{metadata.get('sample_rate', 0)} Hz",
                'file_size': f"{metadata.get('file_size_mb', 0)} MB",
                'bitrate': f"{metadata.get('bitrate_estimate', 0)} kbps"
            }
        }
    
    except Exception as e:
        return {'error': f'Summary generation failed: {str(e)}'}


def safe_analysis(func, *args, **kwargs):
    """Wrapper to catch errors in analysis functions"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return {'error': f'{func.__name__} failed: {str(e)[:100]}'}

def generate_forensic_report(audio_path, output_dir):
    """Generate complete forensic analysis report with error handling"""
    try:
        # Check duration first
        full_duration = librosa.get_duration(path=audio_path)
        
        if full_duration > 300:  # Over 5 minutes
            return {
                'error': 'Audio too long. Maximum duration is 5 minutes.',
                'duration': round(full_duration, 2)
            }
            
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'audio_file': os.path.basename(audio_path),
            'status': 'completed'
        }
        
        # Run all analyses with error handling
        report['metadata'] = safe_analysis(extract_audio_metadata, audio_path)
        report['spectral_analysis'] = safe_analysis(spectral_analysis, audio_path, output_dir)
        report['waveform_analysis'] = safe_analysis(waveform_analysis, audio_path, output_dir)
        report['noise_analysis'] = safe_analysis(noise_analysis, audio_path, output_dir)
        report['tampering_detection'] = safe_analysis(tampering_detection, audio_path, output_dir)
        report['vad'] = safe_analysis(voice_activity_detection, audio_path, output_dir)
        report['enhancement'] = safe_analysis(audio_enhancement, audio_path, output_dir)
        report['noise_floor_analysis'] = safe_analysis(advanced_noise_floor_analysis, audio_path, output_dir)
        report['harmonic_analysis'] = safe_analysis(harmonic_continuity_analysis, audio_path)
        report['breath_analysis'] = safe_analysis(breath_sound_detection, audio_path)
        report['compression_analysis'] = safe_analysis(compression_history_analysis, audio_path)
        report['copy_move_analysis'] = safe_analysis(copy_move_detection, audio_path, output_dir)
        report['splice_detection'] = safe_analysis(precise_splice_detection, audio_path, output_dir)
        
        # Generate summary
        report['user_friendly_summary'] = safe_analysis(generate_user_friendly_summary, report)
        
        # Encode visualizations
        import base64
        viz_files = ['spectrogram.png', 'waveform.png', 'voice_activity.png', 
                     'noise_floor_analysis.png', 'copy_move.png', 'splice_detection.png']
        report['visualizations_base64'] = {}
        
        for viz_file in viz_files:
            viz_path = os.path.join(output_dir, viz_file)
            if os.path.exists(viz_path):
                try:
                    with open(viz_path, 'rb') as f:
                        report['visualizations_base64'][viz_file] = base64.b64encode(f.read()).decode('utf-8')
                except:
                    pass
        
        return report
    
    except Exception as e:
        return {
            'error': f'Analysis failed: {str(e)[:200]}',
            'status': 'failed',
            'timestamp': datetime.now().isoformat()
        }


# ==================== API ENDPOINTS ====================

@app.route('/')
def index():
    return jsonify({
        'service': 'Audio Forensic Analysis API',
        'version': '1.0',
        'description': 'Advanced audio forensic analysis for detecting manipulation and extracting metadata',
        'endpoints': {
            '/analyze': 'POST - Analyze single audio file',
            '/batch': 'POST - Analyze multiple audio files',
            '/compare': 'POST - Compare two audio files',
            '/metadata': 'POST - Extract metadata only',
            '/test': 'GET - Web UI for testing',
            '/health': 'GET - Health check'
        }
    })


@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400
    
    file = request.files['audio']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid format. Supported: MP3, WAV, FLAC, OGG, M4A, AAC'}), 400
    
    file_ext = file.filename.rsplit('.', 1)[1].lower()
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}.{file_ext}")
    
    try:
        file.save(filepath)
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], uuid.uuid4().hex)
        os.makedirs(output_dir, exist_ok=True)
        
        report = generate_forensic_report(filepath, output_dir)
        os.remove(filepath)
        
        # Check if request is from web UI (has format parameter) or from curl/API
        output_format = request.form.get('format', None)
        
        if output_format == 'json':
            # Web UI request - return JSON with base64 images
            response = jsonify(report)
            shutil.rmtree(output_dir)
            return response
        
        else:
            # API/curl request - return ZIP with result.json + PNGs
            # Remove base64 images from JSON
            report_clean = report.copy()
            if 'visualizations_base64' in report_clean:
                del report_clean['visualizations_base64']
            
            # Save as result.json
            report_path = os.path.join(output_dir, 'result.json')
            with open(report_path, 'w') as f:
                json.dump(report_clean, f, indent=2)
            
            # Create ZIP
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.basename(file_path))
            
            zip_buffer.seek(0)
            shutil.rmtree(output_dir)
            
            return send_file(
                zip_buffer,
                as_attachment=True,
                download_name=f'audio_forensic_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip',
                mimetype='application/zip'
            )
        
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500


@app.route('/batch', methods=['POST'])
def batch_analyze():
    if 'audios' not in request.files:
        return jsonify({'error': 'No audio files uploaded'}), 400
    
    files = request.files.getlist('audios')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_output_dir = os.path.join(app.config['OUTPUT_FOLDER'], f'batch_{timestamp}')
    os.makedirs(batch_output_dir, exist_ok=True)
    
    results = []
    
    for file in files:
        if not allowed_file(file.filename):
            continue
        
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}.{file_ext}")
        
        try:
            file.save(filepath)
            file_output_dir = os.path.join(batch_output_dir, file.filename.rsplit('.', 1)[0])
            os.makedirs(file_output_dir, exist_ok=True)
            
            report = generate_forensic_report(filepath, file_output_dir)
            
            report_path = os.path.join(file_output_dir, 'report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            results.append({
                'filename': file.filename,
                'status': 'success',
                'authenticity_score': report['user_friendly_summary']['authenticity_score']
            })
            
            os.remove(filepath)
        
        except Exception as e:
            results.append({'filename': file.filename, 'status': 'failed', 'error': str(e)})
    
    summary = {'total_files': len(files), 'results': results}
    summary_path = os.path.join(batch_output_dir, 'batch_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(batch_output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, batch_output_dir))
    
    zip_buffer.seek(0)
    shutil.rmtree(batch_output_dir)
    
    return send_file(zip_buffer, as_attachment=True, download_name=f'batch_audio_{timestamp}.zip', mimetype='application/zip')


@app.route('/compare', methods=['POST'])
def compare():
    if 'audio1' not in request.files or 'audio2' not in request.files:
        return jsonify({'error': 'Two audio files required'}), 400
    
    file1 = request.files['audio1']
    file2 = request.files['audio2']
    
    filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}.{file1.filename.rsplit('.', 1)[1]}")
    filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}.{file2.filename.rsplit('.', 1)[1]}")
    
    try:
        file1.save(filepath1)
        file2.save(filepath2)
        
        result = compare_audio_files(filepath1, filepath2)
        
        os.remove(filepath1)
        os.remove(filepath2)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/metadata', methods=['POST'])
def metadata_only():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400
    
    file = request.files['audio']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}.{file.filename.rsplit('.', 1)[1]}")
    
    try:
        file.save(filepath)
        metadata = extract_audio_metadata(filepath)
        os.remove(filepath)
        return jsonify(metadata)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/test')
def test_page():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Audio Forensic Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; background: #f5f5f5; }
            h1 { color: #2c3e50; text-align: center; }
            .upload-box { background: white; padding: 30px; border-radius: 8px; text-align: center; margin: 20px 0; }
            input[type="file"] { margin: 15px 0; }
            button { background: #3498db; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background: #2980b9; }
            button:disabled { background: #95a5a6; }
            .loading { display: none; text-align: center; padding: 20px; }
            .loading.show { display: block; }
            .result { display: none; background: white; padding: 25px; border-radius: 8px; margin: 20px 0; }
            .result.show { display: block; }
            .status { font-size: 24px; font-weight: bold; margin: 15px 0; padding: 15px; border-radius: 5px; }
            .low { background: #d4edda; color: #155724; }
            .medium { background: #fff3cd; color: #856404; }
            .high { background: #f8d7da; color: #721c24; }
            .info { margin: 15px 0; line-height: 1.8; }
            .info strong { display: inline-block; width: 180px; }
            select { padding: 8px; margin: 10px; font-size: 14px; }
            .viz-section { margin: 20px 0; text-align: center; }
            .viz-section img { max-width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 10px 0; }
            .section-title { font-size: 18px; font-weight: bold; margin: 20px 0 10px 0; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px; }
            .audio-player { margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>Audio Forensic Analysis</h1>
        
        <div class="upload-box">
            <form id="form">
                <p>Upload an audio file for comprehensive authenticity verification and manipulation detection</p>
                <p style="font-size: 12px; color: #666;">Supported: MP3, WAV, FLAC, OGG, M4A, AAC, WMA</p>
                <input type="file" id="file" accept="audio/*,.mp3,.wav,.flac,.ogg,.m4a,.aac,.wma" required>
                <br>
                <select id="format">
                    <option value="json">Quick Analysis</option>
                    <option value="zip">Full Report (Download)</option>
                </select>
                <br>
                <button type="submit" id="btn">Analyze Audio</button>
            </form>
        </div>
        
        <div class="loading" id="loading">
            <p>⏳ Analyzing audio, please wait...</p>
            <p style="font-size: 12px; color: #666;">This may take a few moments depending on file size</p>
        </div>
        
        <div class="result" id="result">
            <div class="status" id="status"></div>
            <div class="info" id="details"></div>
            <div class="viz-section" id="visualizations"></div>
            <div class="audio-player" id="enhanced"></div>
        </div>
        
        <script>
            document.getElementById('form').onsubmit = async (e) => {
                e.preventDefault();
                
                const formData = new FormData();
                formData.append('audio', document.getElementById('file').files[0]);
                formData.append('format', document.getElementById('format').value);
                
                document.getElementById('loading').classList.add('show');
                document.getElementById('result').classList.remove('show');
                document.getElementById('btn').disabled = true;
                
                try {
                    const res = await fetch('/analyze', { method: 'POST', body: formData });
                    
                    if (res.ok) {
                        const type = res.headers.get('content-type');
                        
                        if (type.includes('json')) {
                            const data = await res.json();
                            const s = data.user_friendly_summary;
                            
                            let statusClass = s.tampering_probability.replace('%','') > 60 ? 'high' : 
                                            s.tampering_probability.replace('%','') > 30 ? 'medium' : 'low';
                            
                            document.getElementById('status').className = 'status ' + statusClass;
                            document.getElementById('status').innerHTML = s.status + '<br>Authenticity Score: ' + s.authenticity_score + '%';
                            
                            let html = '<p><strong>Tampering Risk:</strong> ' + s.tampering_probability + '</p>';
                            
                            html += '<div class="section-title">🎤 Voice Activity Detection</div>';
                            html += '<p><strong>Speakers Detected:</strong> ' + s.speaker_count + ' person(s)</p>';
                            html += '<p><strong>Transcription:</strong> ' + s.transcription + '</p>';
                            html += '<p>' + s.voice_activity_summary + '</p>';
                            
                            html += '<div class="section-title">📊 Spectral Analysis</div>';
                            html += '<p>' + s.spectral_analysis_summary + '</p>';
                            
                            html += '<div class="section-title">✂️ Tampering Detection</div>';
                            html += '<p>' + s.tampering_summary + '</p>';
                            
                            html += '<p><strong>Recommendation:</strong> ' + s.recommendation + '</p>';
                            html += '<hr><p><strong>Format:</strong> ' + s.audio_info.format + '</p>';
                            html += '<p><strong>Duration:</strong> ' + s.audio_info.duration + '</p>';
                            html += '<p><strong>Sample Rate:</strong> ' + s.audio_info.sample_rate + '</p>';
                            html += '<p><strong>Bitrate:</strong> ' + s.audio_info.bitrate + '</p>';
                            html += '<p><strong>File Size:</strong> ' + s.audio_info.file_size + '</p>';
                            
                            document.getElementById('details').innerHTML = html;
                            
                            // Visualizations
                            let vizHtml = '';
                            if (data.visualizations_base64) {
                                vizHtml += '<div class="section-title">📊 Audio Visualizations</div>';
                                
                                if (data.visualizations_base64['waveform.png']) {
                                    vizHtml += '<p><strong>Waveform Analysis</strong></p>';
                                    vizHtml += '<img src="data:image/png;base64,' + data.visualizations_base64['waveform.png'] + '" alt="Waveform">';
                                }
                                
                                if (data.visualizations_base64['spectrogram.png']) {
                                    vizHtml += '<p><strong>Spectrogram</strong></p>';
                                    vizHtml += '<img src="data:image/png;base64,' + data.visualizations_base64['spectrogram.png'] + '" alt="Spectrogram">';
                                }
                                
                                if (data.visualizations_base64['voice_activity.png']) {
                                    vizHtml += '<p><strong>Voice Activity</strong></p>';
                                    vizHtml += '<img src="data:image/png;base64,' + data.visualizations_base64['voice_activity.png'] + '" alt="VAD">';
                                }
                            }
                            document.getElementById('visualizations').innerHTML = vizHtml;
                            
                            document.getElementById('result').classList.add('show');
                        } else {
                            const blob = await res.blob();
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = 'audio_forensic_report.zip';
                            a.click();
                            alert('✓ Full report downloaded');
                        }
                    } else {
                        const error = await res.json();
                        alert('Error: ' + error.error);
                    }
                } catch (err) {
                    console.error('Analysis error:', err);
                    alert('Analysis failed. Please try a different audio file or check your internet connection.');
                } finally {
                    document.getElementById('loading').classList.remove('show');
                    document.getElementById('btn').disabled = false;
                }
            };
        </script>
    </body>
    </html>
    '''


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 200MB', 'code': 'FILE_TOO_LARGE'}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found', 'code': 'NOT_FOUND'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'code': 'INTERNAL_ERROR'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)