import os
import json
import torch
import logging
import traceback
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime
import threading
from collections import defaultdict

import gradio as gr
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from audio_separator.separator import Separator
from audio_separator.separator import architectures

# Helper function to scan the local 'audios' folder
def get_local_audio_files(folder_name="audios"):
    """Scans the local folder for audio files and returns a list of filenames."""
    if not os.path.exists(folder_name):
        try:
            os.makedirs(folder_name)
            print(f"Created directory: {folder_name}")
        except OSError as e:
            print(f"Error creating directory {folder_name}: {e}")
            return []
    
    valid_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.wma', '.aac')
    try:
        files = [f for f in os.listdir(folder_name) if f.lower().endswith(valid_extensions)]
        return sorted(files)
    except Exception as e:
        print(f"Error listing files in {folder_name}: {e}")
        return []

class AudioSeparatorD:
    def __init__(self):
        self.separator = None
        self.available_models = {}
        self.current_model = None
        self.processing_history = []
        self.model_performance_cache = {}
        self.model_recommendations = {}
        self.setup_logging()
        self.model_lock = threading.Lock()
        
    def setup_logging(self):
        """Setup logging for the application"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def get_system_info(self):
        """Get system information for hardware acceleration"""
        info = {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
            "device": "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"),
        }
        
        if torch.cuda.is_available():
            info["memory_total"] = torch.cuda.get_device_properties(0).total_memory
            info["memory_allocated"] = torch.cuda.memory_allocated()
        else:
            info["memory_total"] = 0
            info["memory_allocated"] = 0
            
        return info
    
    def analyze_audio_characteristics(self, audio_file: str) -> Dict:
        """Analyze audio file characteristics for smart model selection"""
        try:
            y, sr = librosa.load(audio_file, sr=None)
            duration = len(y) / sr
            
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            rms = librosa.feature.rms(y=y)[0]
            dynamic_range = np.std(rms)
            
            characteristics = {
                "duration": duration,
                "sample_rate": sr,
                "tempo": float(tempo),
                "avg_spectral_centroid": float(np.mean(spectral_centroids)),
                "avg_spectral_rolloff": float(np.mean(spectral_rolloff)),
                "avg_zero_crossing_rate": float(np.mean(zero_crossing_rate)),
                "dynamic_range": float(dynamic_range),
                "audio_type": self._classify_audio_type(
                    np.mean(spectral_centroids), 
                    float(tempo), 
                    dynamic_range
                )
            }
            
            return characteristics
        except Exception as e:
            self.logger.error(f"Error analyzing audio: {str(e)}")
            return {"audio_type": "unknown", "error": str(e)}
    
    def _classify_audio_type(self, spectral_centroid: float, tempo: float, dynamic_range: float) -> str:
        """Classify audio type based on spectral and temporal features"""
        if spectral_centroid < 1000:
            return "bass_heavy"
        elif spectral_centroid > 4000:
            return "bright_crisp"
        elif tempo > 120:
            return "upbeat"
        elif dynamic_range > 0.1:
            return "dynamic"
        else:
            return "balanced"
    
    def get_available_models(self):
        """Get list of available models with enhanced information"""
        try:
            with self.model_lock:
                if self.separator is None:
                    self.separator = Separator(info_only=True)
                
                models = self.separator.list_supported_model_files()
                simplified_models = self.separator.get_simplified_model_list()
                
                enhanced_models = {}
                for model_name, model_info in simplified_models.items():
                    friendly_name = self._generate_friendly_name(model_name, model_info)
                    use_cases = self._determine_use_cases(model_name, model_info)
                    perf_chars = self._estimate_performance(model_name)
                    
                    enhanced_models[model_name] = {
                        **model_info,
                        "friendly_name": friendly_name,
                        "use_cases": use_cases,
                        "performance_characteristics": perf_chars,
                        "architecture_type": self._get_architecture_type(model_name),
                        "recommended_for": self._get_recommendations(model_name, model_info)
                    }
                
                return enhanced_models
        except Exception as e:
            self.logger.error(f"Error getting available models: {str(e)}")
            return {}
    
    def _generate_friendly_name(self, model_name: str, model_info: Dict) -> str:
        """Generate user-friendly model names"""
        clean_name = model_name.replace('model_', '').replace('.ckpt', '').replace('.yaml', '')
        
        if 'roformer' in model_name.lower():
            return f"🎵 Roformer {clean_name.split('_')[-1] if '_' in clean_name else ''}".strip()
        elif 'demucs' in model_name.lower():
            return f"🥁 Demucs {clean_name.replace('htdemucs', '').replace('_', ' ')}".strip()
        elif 'mdx' in model_name.lower():
            return f"🎤 MDX-Net {clean_name[-3:] if clean_name[-3:].isdigit() else ''}".strip()
        else:
            words = clean_name.replace('_', ' ').split()
            return ' '.join(word.capitalize() for word in words)
    
    def _determine_use_cases(self, model_name: str, model_info: Dict) -> List[str]:
        """Determine what this model is best for"""
        use_cases = []
        
        if 'vocals' in str(model_info).lower():
            use_cases.append("🎤 Vocal Isolation")
        if 'drums' in str(model_info).lower():
            use_cases.append("🥁 Drum Separation")
        if 'bass' in str(model_info).lower():
            use_cases.append("🎸 Bass Extraction")
        if 'instrumental' in str(model_info).lower():
            use_cases.append("🎹 Instrumental")
        if 'guitar' in str(model_info).lower() or 'piano' in str(model_info).lower():
            use_cases.append("🎸 Specific Instruments")
        
        if 'roformer' in model_name.lower():
            use_cases.append("⚡ High Quality")
        elif 'demucs' in model_name.lower():
            use_cases.append("🎛️ Multi-stem")
        elif 'mdx' in model_name.lower():
            use_cases.append("🎵 Fast Processing")
        
        return use_cases[:3]
    
    def _estimate_performance(self, model_name: str) -> Dict:
        """Estimate performance characteristics"""
        perf = {
            "speed_rating": "medium",
            "quality_rating": "medium",
            "memory_usage": "medium"
        }
        
        if 'roformer' in model_name.lower():
            perf.update({"speed_rating": "slow", "quality_rating": "high", "memory_usage": "high"})
        elif 'demucs' in model_name.lower():
            perf.update({"speed_rating": "slow", "quality_rating": "high", "memory_usage": "high"})
        elif 'mdx' in model_name.lower():
            perf.update({"speed_rating": "fast", "quality_rating": "medium", "memory_usage": "low"})
        
        return perf
    
    def _get_architecture_type(self, model_name: str) -> str:
        """Extract architecture type from model name"""
        if 'roformer' in model_name.lower():
            return "🎵 Roformer (MDXC)"
        elif 'demucs' in model_name.lower():
            return "🥁 Demucs"
        elif 'mdx' in model_name.lower():
            return "🎤 MDX-Net"
        elif 'vr' in model_name.lower():
            return "🎛️ VR Arch"
        else:
            return "🔧 Unknown"
    
    def _get_recommendations(self, model_name: str, model_info: Dict) -> Dict:
        """Get specific recommendations for model usage"""
        recommendations = {
            "best_for": "General use",
            "avoid_for": "None",
            "tips": []
        }
        
        if 'roformer' in model_name.lower():
            recommendations.update({
                "best_for": "High-quality vocal isolation",
                "avoid_for": "Real-time processing",
                "tips": ["Best results with longer audio files", "Higher memory usage", "Excellent for final mastering"]
            })
        elif 'demucs' in model_name.lower():
            recommendations.update({
                "best_for": "Multi-stem separation (drums, bass, vocals)",
                "avoid_for": "Simple vocal/instrumental separation",
                "tips": ["Creates multiple output files", "Good for music production", "Slower but comprehensive"]
            })
        elif 'mdx' in model_name.lower():
            recommendations.update({
                "best_for": "Fast vocal isolation",
                "avoid_for": "Multi-instrument separation",
                "tips": ["Quick processing", "Good for demos", "Lower memory requirements"]
            })
        
        return recommendations
    
    def auto_select_model(self, audio_characteristics: Dict, desired_stems: List[str], 
                         priority: str = "quality") -> Optional[str]:
        """Automatically select the best model based on audio characteristics and requirements"""
        try:
            models = self.get_available_models()
            if not models:
                return None
            
            model_scores = {}
            
            for model_name, model_info in models.items():
                score = 0
                perf_chars = model_info.get('performance_characteristics', {})
                
                if priority == "quality":
                    if perf_chars.get('quality_rating') == 'high':
                        score += 10
                    elif perf_chars.get('quality_rating') == 'medium':
                        score += 5
                elif priority == "speed":
                    if perf_chars.get('speed_rating') == 'fast':
                        score += 10
                    elif perf_chars.get('speed_rating') == 'medium':
                        score += 5
                
                audio_type = audio_characteristics.get('audio_type', 'balanced')
                use_cases = model_info.get('use_cases', [])
                
                if audio_type == 'bass_heavy' and '🎸 Bass Extraction' in use_cases:
                    score += 8
                elif audio_type == 'bright_crisp' and '🎤 Vocal Isolation' in use_cases:
                    score += 8
                elif audio_type == 'upbeat' and '🎹 Instrumental' in use_cases:
                    score += 6
                
                model_stems = str(model_info).lower()
                for stem in desired_stems:
                    if stem.lower() in model_stems:
                        score += 5
                
                arch_type = model_info.get('architecture_type', '')
                if priority == "quality" and "Roformer" in arch_type:
                    score += 15
                elif priority == "speed" and "MDX-Net" in arch_type:
                    score += 15
                
                model_scores[model_name] = score
            
            if model_scores:
                best_model = max(model_scores.items(), key=lambda x: x[1])
                return best_model[0]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in auto-select: {str(e)}")
            return None
    
    def compare_models(self, audio_file: str, model_list: List[str]) -> Dict:
        """Enhanced model comparison with detailed metrics"""
        if not audio_file or not model_list:
            return {"error": "Please provide audio file and select models to compare"}
        
        comparison_results = {
            "audio_analysis": self.analyze_audio_characteristics(audio_file),
            "model_results": {},
            "summary": {},
            "recommendations": []
        }
        
        for model_name in model_list:
            try:
                start_time = time.time()
                success, message = self.initialize_separator(model_name)
                
                if not success:
                    comparison_results["model_results"][model_name] = {
                        "status": "Failed",
                        "error": message,
                        "processing_time": 0
                    }
                    continue
                
                output_files = self.separator.separate(audio_file)
                processing_time = time.time() - start_time
                
                if output_files and os.path.exists(output_files[0]):
                    audio_data, sample_rate = sf.read(output_files[0])
                    quality_metrics = self._calculate_quality_metrics(audio_data, sample_rate)
                    
                    comparison_results["model_results"][model_name] = {
                        "status": "Success",
                        "processing_time": processing_time,
                        "output_files": len(output_files),
                        "sample_rate": sample_rate,
                        "duration": len(audio_data) / sample_rate,
                        "quality_metrics": quality_metrics,
                        "output_stems": [os.path.basename(f) for f in output_files],
                        "model_info": self.get_available_models().get(model_name, {})
                    }
                    
                    for file_path in output_files:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                else:
                    comparison_results["model_results"][model_name] = {
                        "status": "Failed",
                        "error": "No output files generated",
                        "processing_time": processing_time
                    }
                    
            except Exception as e:
                comparison_results["model_results"][model_name] = {
                    "status": "Error",
                    "error": str(e),
                    "processing_time": 0
                }
        
        comparison_results["summary"] = self._generate_comparison_summary(comparison_results["model_results"])
        comparison_results["recommendations"] = self._generate_recommendations(
            comparison_results["audio_analysis"], 
            comparison_results["model_results"]
        )
        
        return comparison_results
    
    def _calculate_quality_metrics(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Calculate audio quality metrics"""
        try:
            rms = np.sqrt(np.mean(audio_data**2))
            peak = np.max(np.abs(audio_data))
            dynamic_range = 20 * np.log10(peak / (rms + 1e-10))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate))
            
            return {
                "rms_level": float(rms),
                "peak_level": float(peak),
                "dynamic_range": float(dynamic_range),
                "spectral_centroid": float(spectral_centroid),
                "length_samples": len(audio_data),
                "length_seconds": len(audio_data) / sample_rate
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _generate_comparison_summary(self, model_results: Dict) -> Dict:
        """Generate summary statistics from model comparison"""
        successful_results = {k: v for k, v in model_results.items() if v.get("status") == "Success"}
        
        if not successful_results:
            return {"message": "No successful model runs to compare"}
        
        summary = {
            "total_models": len(model_results),
            "successful_models": len(successful_results),
            "fastest_model": None,
            "slowest_model": None,
            "best_quality": None,
            "average_processing_time": 0
        }
        
        if successful_results:
            times = {k: v.get("processing_time", 0) for k, v in successful_results.items()}
            summary["fastest_model"] = min(times.items(), key=lambda x: x[1])[0]
            summary["slowest_model"] = max(times.items(), key=lambda x: x[1])[0]
            summary["average_processing_time"] = np.mean(list(times.values()))
        
        return summary
    
    def _generate_recommendations(self, audio_analysis: Dict, model_results: Dict) -> List[str]:
        """Generate intelligent recommendations based on comparison"""
        recommendations = []
        
        successful_models = {k: v for k, v in model_results.items() if v.get("status") == "Success"}
        
        if successful_models:
            fastest_model = min(successful_models.items(), 
                              key=lambda x: x[1].get("processing_time", float('inf')))
            recommendations.append(f"⚡ Fastest: {fastest_model[0]} ({fastest_model[1]['processing_time']:.2f}s)")
            
            most_outputs = max(successful_models.items(), 
                             key=lambda x: x[1].get("output_files", 0))
            recommendations.append(f"🎛️ Most stems: {most_outputs[0]} ({most_outputs[1]['output_files']} files)")
        
        audio_type = audio_analysis.get('audio_type', 'unknown')
        if audio_type == 'bass_heavy':
            recommendations.append("🎸 Consider models with bass separation capabilities")
        elif audio_type == 'bright_crisp':
            recommendations.append("🎤 Models optimized for vocal clarity work best")
        elif audio_type == 'upbeat':
            recommendations.append("🎹 Fast processing models recommended for energetic tracks")
        
        return recommendations
    
    def initialize_separator(self, model_name: str = None, **kwargs):
        """Initialize the separator with specified parameters"""
        try:
            with self.model_lock:
                if self.separator is not None:
                    del self.separator
                    torch.cuda.empty_cache()
                
                if model_name is None:
                    models = self.get_available_models()
                    if models:
                        model_name = list(models.keys())[0]
                    else:
                        return False, "No models available"
                
                self.separator = Separator(
                    output_format="WAV",
                    use_autocast=True,
                    use_soundfile=True,
                    **kwargs
                )
                
                self.separator.load_model(model_name)
                self.current_model = model_name
                
                return True, f"Successfully initialized with model: {model_name}"
                
        except Exception as e:
            self.logger.error(f"Error initializing separator: {str(e)}")
            return False, f"Error initializing separator: {str(e)}"
    
    def infer(self, audio_file: str, model_name: str, output_format: str = "WAV", 
                             quality_preset: str = "Standard", custom_params: Dict = None,
                             enable_auto_optimize: bool = True):
        """Enhanced audio processing with auto-optimization"""
        if audio_file is None:
            return None, "No audio file provided"
        
        if model_name is None:
            return None, "No model selected"
        
        if enable_auto_optimize:
            audio_analysis = self.analyze_audio_characteristics(audio_file)
            custom_params = self._optimize_parameters_for_audio(audio_analysis, custom_params)
        
        if self.separator is None or self.current_model != model_name:
            success, message = self.initialize_separator(model_name)
            if not success:
                return None, message
        
        try:
            start_time = time.time()
            
            if custom_params is None:
                custom_params = {}
            
            if quality_preset == "Fast":
                custom_params.update({
                    "mdx_params": {"batch_size": 4, "overlap": 0.1, "segment_size": 128},
                    "vr_params": {"batch_size": 8, "aggression": 3},
                    "demucs_params": {"shifts": 1, "overlap": 0.1},
                    "mdxc_params": {"batch_size": 4, "overlap": 4}
                })
            elif quality_preset == "High Quality":
                custom_params.update({
                    "mdx_params": {"batch_size": 1, "overlap": 0.5, "segment_size": 512, "enable_denoise": True},
                    "vr_params": {"batch_size": 1, "aggression": 8, "enable_tta": True, "enable_post_process": True},
                    "demucs_params": {"shifts": 4, "overlap": 0.5, "segments_enabled": False},
                    "mdxc_params": {"batch_size": 1, "overlap": 16, "pitch_shift": 0}
                })
            
            for key, value in custom_params.items():
                if hasattr(self.separator, key):
                    setattr(self.separator, key, value)
            
            output_files = self.separator.separate(audio_file)
            
            processing_time = time.time() - start_time
            
            output_audio = {}
            for file_path in output_files:
                if os.path.exists(file_path):
                    stem_name = os.path.splitext(os.path.basename(file_path))[0]
                    audio_data, sample_rate = sf.read(file_path)
                    output_audio[stem_name] = (sample_rate, audio_data)
                    os.remove(file_path)
            
            if not output_audio:
                return None, "No output files generated"
            
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "processing_time": processing_time,
                "output_files": list(output_audio.keys()),
                "audio_analysis": self.analyze_audio_characteristics(audio_file) if enable_auto_optimize else {},
                "quality_preset": quality_preset
            }
            self.processing_history.append(history_entry)
            
            return output_audio, f"Processing completed in {processing_time:.2f}s with model: {model_name}"
            
        except Exception as e:
            error_msg = f"Error processing audio: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return None, error_msg
    
    def _optimize_parameters_for_audio(self, audio_analysis: Dict, custom_params: Dict) -> Dict:
        """Automatically optimize parameters based on audio characteristics"""
        if custom_params is None:
            custom_params = {}
        
        duration = audio_analysis.get('duration', 0)
        audio_type = audio_analysis.get('audio_type', 'balanced')
        
        if duration > 300:
            custom_params.setdefault('mdx_params', {})['batch_size'] = 2
            custom_params.setdefault('vr_params', {})['batch_size'] = 2
        
        if audio_type == 'bass_heavy':
            custom_params.setdefault('vr_params', {})['aggression'] = 7
        
        if audio_type == 'bright_crisp':
            custom_params.setdefault('vr_params', {})['enable_post_process'] = True
        
        if audio_analysis.get('dynamic_range', 0) > 0.1:
            custom_params.setdefault('vr_params', {})['enable_tta'] = True
        
        return custom_params
    
    def get_phistory(self):
        """Get enhanced processing history with analytics"""
        if not self.processing_history:
            return "No processing history available"
        
        history_text = "🎵 Enhanced Processing History\n\n"
        
        for i, entry in enumerate(self.processing_history[-10:], 1):
            history_text += f"**{i}. {entry['timestamp'][:19]}**\n"
            history_text += f"   Model: {entry['model']}\n"
            history_text += f"   Time: {entry['processing_time']:.2f}s\n"
            history_text += f"   Stems: {', '.join(entry['output_files'])}\n"
            
            if 'audio_analysis' in entry and entry['audio_analysis']:
                audio_type = entry['audio_analysis'].get('audio_type', 'unknown')
                duration = entry['audio_analysis'].get('duration', 0)
                history_text += f"   Audio: {audio_type} ({duration:.1f}s)\n"
            
            if 'quality_preset' in entry:
                history_text += f"   Preset: {entry['quality_preset']}\n"
            
            history_text += "\n"
        
        return history_text
    
    def reset_history(self):
        """Reset processing history"""
        self.processing_history = []
        return "Processing history cleared"


# Initialize the enhanced demo
demo1 = AudioSeparatorD()

# Create the Gradio interface directly
with gr.Blocks(theme="NeoPy/Soft", title="🎵 Enhanced Audio Separator") as app:
    gr.Markdown(
        """
        # 🎵 Audio Separator Web UI
        
        **Smart AI-Powered Audio Source Separation with Auto-Selection & Advanced Model Comparison**
        
        ✨ **Features**: Upload or select from 'audios' folder, auto model selection, performance analytics, smart parameter optimization.
        """
    )
    
    # System Information
    with gr.Accordion("🖥️ System Information", open=False):
        system_info = demo1.get_system_info()
        info_text = f"""
        **PyTorch Version:** {system_info['pytorch_version']}
        **Hardware Acceleration:** {system_info['device'].upper()}
        **CUDA Available:** {system_info['cuda_available']} (Version: {system_info['cuda_version']})
        **Apple Silicon (MPS):** {system_info['mps_available']}
        **GPU Memory:** {system_info['memory_allocated'] // 1024**2}MB / {system_info['memory_total'] // 1024**2}MB
        """
        gr.Markdown(info_text)
    
    with gr.Row():
        with gr.Column():
            # --- Input Section ---
            gr.Markdown("### 🎵 Input Source")
            
            # Tab for Upload vs Folder
            with gr.Tabs():
                with gr.Tab("Upload File"):
                    audio_input = gr.Audio(
                        label="Upload Audio File",
                        type="filepath"
                    )
                
                with gr.Tab("Select from Folder"):
                    folder_audio_choice = gr.Dropdown(
                        label="Select from 'audios' folder",
                        choices=get_local_audio_files(),
                        interactive=True
                    )
                    refresh_folder_btn = gr.Button("🔄 Scan 'audios' Folder", size="sm")
                    gr.Markdown("*Place your audio files in the 'audios' folder in the same directory as this script.*")

            # Auto-analyze button
            analyze_btn = gr.Button("🔍 Analyze Audio", variant="secondary")
            
            # Audio analysis output
            audio_analysis_output = gr.JSON(label="Audio Analysis Results", visible=False)
            
            # Enhanced model selection
            model_list = demo1.get_available_models()
            
            # Model dropdown with enhanced display
            model_dropdown = gr.Dropdown(
                choices=list(model_list.keys()) if model_list else [],
                value=list(model_list.keys())[0] if model_list else None,
                label="🤖 AI Model Selection",
                elem_id="model_dropdown"
            )
            
            # Add info text separately
            gr.Markdown("*Choose an AI model or use auto-selection*")
            
            # Auto-selection controls
            with gr.Row():
                auto_select_btn = gr.Button("🎯 Auto-Select Best Model", variant="primary")
                priority_radio = gr.Radio(
                    choices=["Quality", "Speed", "Balanced"],
                    value="Quality",
                    label="Selection Priority"
                )
            
            gr.Markdown("*What matters most for model selection?*")
            
            # Model info display
            model_info_display = gr.JSON(label="📊 Selected Model Information")
            
            # Quality preset and optimization
            with gr.Row():
                quality_preset = gr.Radio(
                    choices=["Fast", "Standard", "High Quality", "Custom"],
                    value="Standard",
                    label="⚡ Processing Quality"
                )
                
                auto_optimize = gr.Checkbox(
                    label="🧠 Auto-Optimize Parameters",
                    value=True
                )
            
            gr.Markdown("*Automatically optimize parameters based on audio analysis*")
            
            # Enhanced advanced parameters
            with gr.Accordion("🔧 Advanced Parameters", open=False):
                with gr.Row():
                    batch_size = gr.Slider(1, 8, value=1, step=1, label="Batch Size")
                    segment_size = gr.Slider(64, 1024, value=256, step=64, label="Segment Size")
                    overlap = gr.Slider(0.1, 0.5, value=0.25, step=0.05, label="Overlap")
                
                with gr.Row():
                    denoise = gr.Checkbox(label="Enable Denoise", value=False)
                    tta = gr.Checkbox(label="Enable TTA", value=False)
                    post_process = gr.Checkbox(label="Enable Post-Processing", value=False)
                    pitch_shift = gr.Slider(-12, 12, value=0, step=1, label="Pitch Shift (semitones)")
            
            # Process button
            process_btn = gr.Button("🎵 Smart Separate Audio", variant="primary", size="lg")
        
        with gr.Column():
            # Status and results
            status_output = gr.Textbox(label="📋 Status", lines=4)
            
            # Enhanced output tabs
            with gr.Tabs():
                with gr.Tab("🎤 Vocals"):
                    vocals_output = gr.Audio(label="Vocals")
                
                with gr.Tab("🎹 Instrumental"):
                    instrumental_output = gr.Audio(label="Instrumental")
                
                with gr.Tab("🥁 Drums"):
                    drums_output = gr.Audio(label="Drums")
                
                with gr.Tab("🎸 Bass"):
                    bass_output = gr.Audio(label="Bass")
                
                with gr.Tab("🎛️ Other Stems"):
                    other_output = gr.Audio(label="Other Stems")
            
            # Performance metrics
            performance_metrics = gr.JSON(label="📈 Performance Metrics", visible=False)
            
            # Download section
            with gr.Accordion("📥 Batch & Download", open=False):
                gr.Markdown("### 🔄 Batch Processing")
                batch_files = gr.File(
                    file_count="multiple", 
                    file_types=[".wav", ".mp3", ".flac", ".m4a"], 
                    label="Batch Audio Files"
                )
                
                with gr.Row():
                    batch_btn = gr.Button("⚡ Process Batch")
                    auto_batch_btn = gr.Button("🎯 Auto-Select & Batch")
                
                batch_output = gr.File(label="📦 Download Batch Results")
    
    # Enhanced Model Management Tabs
    with gr.Tabs():
        with gr.Tab("🔍 Model Explorer"):
            gr.Markdown("## 🧠 Intelligent Model Comparison & Selection")
            
            model_info = gr.JSON(value=demo1.get_available_models(), label="📊 Model Database")
            refresh_models_btn = gr.Button("🔄 Refresh Models")
            
            with gr.Row():
                filter_architecture = gr.Dropdown(
                    choices=["All", "MDX-Net", "Demucs", "Roformer", "VR Arch"],
                    value="All",
                    label="Filter by Architecture"
                )
                filter_use_case = gr.Dropdown(
                    choices=["All", "Vocals", "Instrumental", "Drums", "Bass", "Multi-stem"],
                    value="All",
                    label="Filter by Use Case"
                )
                filter_priority = gr.Dropdown(
                    choices=["All", "Quality", "Speed", "Memory Efficient"],
                    value="All",
                    label="Filter by Priority"
                )
            
            filtered_models = gr.Dropdown(
                choices=list(model_list.keys())[:10] if model_list else [],
                multiselect=True,
                label="🎯 Models for Comparison"
            )
            
            gr.Markdown("*Select up to 5 models for detailed comparison*")
            
            compare_btn = gr.Button("🔬 Advanced Model Comparison")
            comparison_results = gr.JSON(label="📊 Comparison Results")
        
        with gr.Tab("📈 Analytics & History"):
            history_output = gr.Textbox(label="📜 Processing History", lines=15)
            
            with gr.Row():
                refresh_history_btn = gr.Button("🔄 Refresh History")
                reset_history_btn = gr.Button("🗑️ Clear History", variant="stop")
                export_history_btn = gr.Button("📊 Export Analytics")
            
            analytics_output = gr.JSON(label="📊 Analytics Dashboard")
        
        with gr.Tab("🎯 Smart Recommendations"):
            gr.Markdown("## 🤖 AI-Powered Model Recommendations")
            
            recommendation_status = gr.Textbox(label="Recommendation Status", lines=3)
            
            with gr.Row():
                get_recommendations_btn = gr.Button("🎯 Get Smart Recommendations")
                apply_recommendation_btn = gr.Button("✨ Apply Best Recommendation")
            
            recommendations_display = gr.JSON(label="🎯 Personalized Recommendations")

    # --- Helper Function for UI Logic ---
    def get_effective_file(uploaded_file, folder_file):
        """Determines which file to use: uploaded one or one from the 'audios' folder."""
        if uploaded_file is not None:
            return uploaded_file
        if folder_file:
            return os.path.join("audios", folder_file)
        return None

    # Event handlers
    def analyze_audio_wrapper(uploaded_file, folder_file):
        effective_file = get_effective_file(uploaded_file, folder_file)
        
        if not effective_file:
            return None, "No audio file provided"
        
        analysis = demo1.analyze_audio_characteristics(effective_file)
        
        if "error" not in analysis:
            formatted_analysis = f"""
            **Audio Type:** {analysis.get('audio_type', 'Unknown').title().replace('_', ' ')}
            **Duration:** {analysis.get('duration', 0):.1f} seconds
            **Sample Rate:** {analysis.get('sample_rate', 0)} Hz
            **Tempo:** {analysis.get('tempo', 0):.1f} BPM
            **Spectral Characteristics:** {analysis.get('avg_spectral_centroid', 0):.0f} Hz (centroid)
            **Dynamic Range:** {analysis.get('dynamic_range', 0):.3f}
            """
            return analysis, formatted_analysis
        else:
            return analysis, f"Analysis failed: {analysis['error']}"
    
    def auto_select_model_wrapper(uploaded_file, folder_file, priority):
        effective_file = get_effective_file(uploaded_file, folder_file)
        
        if not effective_file:
            return None, "No audio file provided", None
        
        audio_analysis = demo1.analyze_audio_characteristics(effective_file)
        
        desired_stems = ["vocals"]
        if audio_analysis.get('audio_type') == 'bass_heavy':
            desired_stems.append("bass")
        elif audio_analysis.get('tempo', 0) > 120:
            desired_stems.append("drums")
        
        selected_model = demo1.auto_select_model(
            audio_analysis, desired_stems, priority.lower()
        )
        
        if selected_model:
            models = demo1.get_available_models()
            model_info = models.get(selected_model, {})
            
            return (
                selected_model,
                f"🎯 Auto-selected: {model_info.get('friendly_name', selected_model)}\n"
                f"Architecture: {model_info.get('architecture_type', 'Unknown')}\n"
                f"Best for: {', '.join(model_info.get('use_cases', [])[:2])}",
                model_info
            )
        else:
            return None, "Auto-selection failed - no suitable model found", None
    
    def update_model_info(model_name):
        if not model_name:
            return None
        
        models = demo1.get_available_models()
        model_info = models.get(model_name, {})
        
        if model_info:
            friendly_info = {
                "🤖 Friendly Name": model_info.get('friendly_name', model_name),
                "🏗️ Architecture": model_info.get('architecture_type', 'Unknown'),
                "💡 Best For": model_info.get('use_cases', []),
                "⚡ Performance": model_info.get('performance_characteristics', {}),
                "🎯 Recommendations": model_info.get('recommended_for', {}),
                "📊 Technical Details": {
                    "Filename": model_name,
                    "Supported Stems": len(str(model_info)) // 10
                }
            }
            return friendly_info
        
        return {"error": "Model information not available"}
    
    def infer_wrapper(uploaded_file, folder_file, model_name, quality_preset, batch_size, segment_size, 
                             overlap, denoise, tta, post_process, pitch_shift, auto_optimize):
        
        effective_file = get_effective_file(uploaded_file, folder_file)
        
        if not effective_file or not model_name:
            return None, None, None, None, None, "Please select an audio file (upload or from folder) and a model", None
        
        custom_params = {
            "mdx_params": {
                "batch_size": int(batch_size),
                "segment_size": int(segment_size),
                "overlap": float(overlap),
                "enable_denoise": denoise
            },
            "vr_params": {
                "batch_size": int(batch_size),
                "enable_tta": tta,
                "enable_post_process": post_process,
                "aggression": 5
            },
            "demucs_params": {
                "overlap": float(overlap)
            },
            "mdxc_params": {
                "batch_size": int(batch_size),
                "overlap": int(overlap * 10),
                "pitch_shift": int(pitch_shift)
            }
        }
        
        output_audio, status = demo1.infer(
            effective_file, model_name, 
            quality_preset=quality_preset, 
            custom_params=custom_params,
            enable_auto_optimize=auto_optimize
        )
        
        if output_audio is None:
            return None, None, None, None, None, status, None
        
        vocals = None
        instrumental = None
        drums = None
        bass = None
        other = None
        
        for stem_name, (sample_rate, audio_data) in output_audio.items():
            if "vocal" in stem_name.lower():
                vocals = (sample_rate, audio_data)
            elif "instrumental" in stem_name.lower():
                instrumental = (sample_rate, audio_data)
            elif "drum" in stem_name.lower():
                drums = (sample_rate, audio_data)
            elif "bass" in stem_name.lower():
                bass = (sample_rate, audio_data)
            else:
                other = (sample_rate, audio_data)
        
        performance_metrics = {
            "Model": model_name,
            "Quality Preset": quality_preset,
            "Output Stems": len(output_audio),
            "Processing": "Completed Successfully"
        }
        
        return vocals, instrumental, drums, bass, other, status, performance_metrics
    
    def compare_models_advanced(uploaded_file, folder_file, model_list):
        effective_file = get_effective_file(uploaded_file, folder_file)
        
        if not effective_file or not model_list:
            return {"error": "Please provide audio file and select models to compare"}
        
        results = demo1.compare_models(effective_file, model_list)
        return results
    
    def get_smart_recommendations_wrapper(uploaded_file, folder_file):
        effective_file = get_effective_file(uploaded_file, folder_file)
        
        if not effective_file:
            return "Please upload or select an audio file first", {}
        
        audio_analysis = demo1.analyze_audio_characteristics(effective_file)
        models = demo1.get_available_models()
        
        recommendations = {
            "audio_analysis": audio_analysis,
            "recommended_models": [],
            "tips": []
        }
        
        quality_models = []
        speed_models = []
        
        for model_name, model_info in models.items():
            perf_chars = model_info.get('performance_characteristics', {})
            
            if perf_chars.get('quality_rating') == 'high':
                quality_models.append({
                    'model': model_name,
                    'name': model_info.get('friendly_name', model_name),
                    'reason': 'High quality output'
                })
            
            if perf_chars.get('speed_rating') == 'fast':
                speed_models.append({
                    'model': model_name,
                    'name': model_info.get('friendly_name', model_name),
                    'reason': 'Fast processing'
                })
        
        recommendations["recommended_models"] = {
            "🎯 For Best Quality": quality_models[:3],
            "⚡ For Speed": speed_models[:3]
        }
        
        audio_type = audio_analysis.get('audio_type', 'balanced')
        if audio_type == 'bass_heavy':
            recommendations["tips"].append("🎸 Models with bass separation work best")
        elif audio_type == 'bright_crisp':
            recommendations["tips"].append("🎤 Post-processing enabled for vocal clarity")
        elif audio_type == 'upbeat':
            recommendations["tips"].append("🥁 Consider drum isolation for energetic tracks")
        
        status = f"✅ Generated recommendations for {audio_analysis.get('audio_type', 'unknown')} audio"
        return status, recommendations
    
    def apply_best_recommendation_wrapper(uploaded_file, folder_file):
        effective_file = get_effective_file(uploaded_file, folder_file)
        
        if not effective_file:
            return None, "Please upload or select an audio file first", None
        
        audio_analysis = demo1.analyze_audio_characteristics(effective_file)
        selected_model = demo1.auto_select_model(
            audio_analysis, ["vocals"], "quality"
        )
        
        if selected_model:
            models = demo1.get_available_models()
            model_info = models.get(selected_model, {})
            
            return (
                selected_model,
                f"✨ Applied recommendation: {model_info.get('friendly_name', selected_model)}",
                model_info
            )
        else:
            return None, "Could not generate recommendations", None
    
    # Wire up event handlers
    analyze_btn.click(
        fn=analyze_audio_wrapper,
        inputs=[audio_input, folder_audio_choice],
        outputs=[audio_analysis_output, recommendation_status]
    )
    
    auto_select_btn.click(
        fn=auto_select_model_wrapper,
        inputs=[audio_input, folder_audio_choice, priority_radio],
        outputs=[model_dropdown, recommendation_status, model_info_display]
    )
    
    model_dropdown.change(
        fn=update_model_info,
        inputs=[model_dropdown],
        outputs=[model_info_display]
    )
    
    process_btn.click(
        fn=infer_wrapper,
        inputs=[
            audio_input, folder_audio_choice, model_dropdown, quality_preset,
            batch_size, segment_size, overlap, denoise, tta, post_process, 
            pitch_shift, auto_optimize
        ],
        outputs=[
            vocals_output, instrumental_output, drums_output, 
            bass_output, other_output, status_output, performance_metrics
        ]
    )
    
    compare_btn.click(
        fn=compare_models_advanced,
        inputs=[audio_input, folder_audio_choice, filtered_models],
        outputs=[comparison_results]
    )
    
    refresh_folder_btn.click(
        fn=lambda: gr.Dropdown(choices=get_local_audio_files(), value=None),
        outputs=[folder_audio_choice]
    )
    
    refresh_models_btn.click(
        fn=lambda: demo1.get_available_models(),
        outputs=[model_info]
    )
    
    refresh_history_btn.click(
        fn=lambda: demo1.get_phistory(),
        outputs=[history_output]
    )
    
    reset_history_btn.click(
        fn=lambda: demo1.reset_history(),
        outputs=[history_output]
    )
    
    get_recommendations_btn.click(
        fn=get_smart_recommendations_wrapper,
        inputs=[audio_input, folder_audio_choice],
        outputs=[recommendation_status, recommendations_display]
    )
    
    apply_recommendation_btn.click(
        fn=apply_best_recommendation_wrapper,
        inputs=[audio_input, folder_audio_choice],
        outputs=[model_dropdown, recommendation_status, model_info_display]
    )
    
    # Batch processing
    def batch_inf(batch_files, model_name):
        if not batch_files or not model_name:
            return None, "Please upload batch files and select a model"
        
        import zipfile
        import io
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_info in batch_files:
                output_audio, _ = demo1.infer(file_info, model_name)
                if output_audio is not None:
                    for stem_name, (sample_rate, audio_data) in output_audio.items():
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                            sf.write(tmp_file.name, audio_data, sample_rate)
                            with open(tmp_file.name, 'rb') as f:
                                zip_file.writestr(f"{os.path.splitext(os.path.basename(file_info))[0]}_{stem_name}.wav", f.read())
                            os.unlink(tmp_file.name)
        
        zip_buffer.seek(0)
        return gr.File(value=zip_buffer, visible=True), f"Batch processing completed for {len(batch_files)} files"
    
    batch_btn.click(
        fn=batch_inf,
        inputs=[batch_files, model_dropdown],
        outputs=[batch_output, status_output]
    )
    
    def auto_batch_process(batch_files, priority):
        if not batch_files:
            return None, "Please upload batch files"
        
        if batch_files:
            audio_analysis = demo1.analyze_audio_characteristics(batch_files[0])
            selected_model = demo1.auto_select_model(audio_analysis, ["vocals"], priority.lower())
            
            if selected_model:
                return batch_inf(batch_files, selected_model)
        
        return None, "Auto-selection failed"
    
    auto_batch_btn.click(
        fn=auto_batch_process,
        inputs=[batch_files, priority_radio],
        outputs=[batch_output, status_output]
    )

app.launch(
    server_port=7860,
    share=True,
    ssr_mode=True
)
