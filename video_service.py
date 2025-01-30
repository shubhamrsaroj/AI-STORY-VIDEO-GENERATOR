import sys
from pathlib import Path
import logging
import os
import time
from typing import List, Dict, Tuple
import cv2
import json
import shutil
import subprocess
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import clip
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPProcessor, CLIPModel
import faiss
from audiocraft.models import MusicGen
from ultralytics import YOLO
import ffmpeg
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx, CompositeVideoClip
from moviepy.video.fx.all import *
import tempfile
import imageio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define directory constants at the top level
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
TEMP_FOLDER = os.path.join(BASE_DIR, 'temp')
CLIPS_FOLDER = os.path.join(BASE_DIR, 'clips')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')

# Create all necessary directories
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, TEMP_FOLDER, CLIPS_FOLDER, MODELS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Update the path to categories file
MODELS_DIR = os.path.join(MODELS_FOLDER, 'models')
CATEGORIES_FILE = os.path.join(MODELS_DIR, 'categories_places365.txt')

# Load categories
try:
    with open(CATEGORIES_FILE) as f:
        CATEGORIES_PLACES365 = [line.strip().split(' ')[0][3:] for line in f]
except FileNotFoundError:
    logger.error(f"Categories file not found at {CATEGORIES_FILE}")
    CATEGORIES_PLACES365 = []

# Define constants
MODEL_ID = "openai/clip-vit-base-patch32"

class Places365Model(torch.nn.Module):
    def __init__(self):
        super(Places365Model, self).__init__()
        import torch.nn as nn
        
        # Update paths to use server/models
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(current_dir, 'models')  # This will point to server/models
        self.model_path = os.path.join(self.model_dir, 'resnet50_places365.pth.tar')
        self.categories_path = os.path.join(self.model_dir, 'categories_places365.txt')
        
        # Check if model exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Places365 model not found at {self.model_path}. "
                "Please check the models are in the server/models directory!"
            )
        
        # Load the pre-trained model
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        # Replace the last layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 365)
        
        # Load Places365 weights
        state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
        
        # Fix the state dict keys
        new_state_dict = {}
        for k, v in state_dict['state_dict'].items():
            name = k.replace('module.', '')  # Remove 'module.' prefix
            new_state_dict[name] = v
            
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        
        # Define image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        return self.model(x)

class VideoUnderstandingService:
    def __init__(self):
        """Initialize all AI models"""
        try:
            # Set device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Initialize YOLO
            self.yolo = YOLO("yolov8n.pt")
            logger.info("✓ YOLO initialized")
            
            # Initialize Places365
            self.places365 = Places365Model().to(self.device)
            logger.info("✓ Places365 initialized")
            
            # Initialize CLIP
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            logger.info("✓ CLIP initialized")
            
            # Initialize Story Generator (using OPT instead of Mistral)
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
            self.story_model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
            self.story_model.to(self.device)
            logger.info("✓ Story Generator initialized")
            
            # Initialize MusicGen
            self.music_model = MusicGen.get_pretrained('facebook/musicgen-small')
            self.music_model.set_generation_params(duration=8)
            logger.info("✓ MusicGen initialized")
            
            # Initialize FAISS
            self.clip_dimension = 512
            self.faiss_index = faiss.IndexFlatIP(self.clip_dimension)
            logger.info("✓ FAISS initialized")
            
            # Add output directories
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
            self.output_dir = os.path.join(self.base_dir, 'output')
            self.clips_dir = os.path.join(self.base_dir, 'clips')
            self.temp_dir = os.path.join(self.base_dir, 'temp')
            
            # Create directories if they don't exist
            for directory in [self.output_dir, self.clips_dir, self.temp_dir]:
                os.makedirs(directory, exist_ok=True)
            
            logger.info(f"Videos will be saved to: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            raise

    def analyze_frame(self, frame) -> Dict:
        """Analyze frame using YOLO and Places365"""
        try:
            # YOLO detection
            yolo_results = self.yolo.predict(frame, conf=0.3)
            detected_objects = []
            
            if yolo_results and len(yolo_results[0].boxes) > 0:
                for box in yolo_results[0].boxes:
                    cls = int(box.cls[0])
                    name = yolo_results[0].names[cls]
                    detected_objects.append(name)
            
            # Places365 scene recognition
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = self.places365.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.places365(input_tensor)
                # Get top 5 predictions
                _, pred_idx = output.data.topk(5, 1)
                pred_idx = pred_idx[0].tolist()
                
            # Get scene categories
            scene_categories = [CATEGORIES_PLACES365[idx] for idx in pred_idx]
            
            return {
                'objects': detected_objects,
                'scenes': scene_categories
            }
            
        except Exception as e:
            logger.error(f"Frame analysis error: {str(e)}")
            return {'objects': [], 'scenes': []}

    def detect_scenes(self, video_paths: List[str]) -> List[Dict]:
        """Detect scenes in video using YOLO and Places365"""
        try:
            all_scenes = []
            
            for video_path in video_paths:
                try:
                    if not os.path.exists(video_path):
                        logger.error(f"Video file not found: {video_path}")
                        continue
                        
                    logger.info(f"Processing video: {video_path}")
                    
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        logger.error(f"Failed to open video: {video_path}")
                        continue
                        
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    video_scenes = []
                    frame_count = 0
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        # Calculate current timestamp
                        timestamp = frame_count / fps
                        
                        # Process every second
                        if frame_count % fps == 0:
                            try:
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                results = self.yolo.predict(
                                    frame_rgb,
                                    conf=0.2,
                                    iou=0.45,
                                    max_det=20
                                )
                                
                                detected_objects = []
                                if len(results) > 0 and len(results[0].boxes) > 0:
                                    for box in results[0].boxes:
                                        if box.conf is not None and box.conf[0] > 0.2:
                                            cls = int(box.cls[0])
                                            name = results[0].names[cls]
                                            detected_objects.append({
                                                'name': name,
                                                'confidence': float(box.conf[0])
                                            })
                                            
                                if detected_objects:
                                    video_scenes.append({
                                        'timestamp': timestamp,
                                        'frame_number': frame_count,
                                        'objects': detected_objects
                                    })
                                    logger.info(f"Frame {frame_count}: Found {len(detected_objects)} objects at {timestamp:.2f}s")
                                    
                            except Exception as e:
                                logger.warning(f"Frame processing error: {str(e)}")
                                continue
                                
                        frame_count += 1
                        
                    cap.release()
                    
                    if video_scenes:
                        video_scenes.sort(key=lambda x: x['timestamp'])
                        all_scenes.extend(video_scenes)
                        logger.info(f"Found {len(video_scenes)} scenes in {video_path}")
                    else:
                        logger.warning(f"No scenes detected in {video_path}")
                        
                except Exception as e:
                    logger.error(f"Error processing video {video_path}: {str(e)}")
                    continue
            
            return all_scenes
            
        except Exception as e:
            logger.error(f"Scene detection error: {str(e)}")
            return []

    def generate_story(self, prompt: str, scenes: List[Dict]) -> List[str]:
        """Generate story using OPT model"""
        try:
            # Extract objects and scenes as strings, not dicts
            objects_context = set()
            scenes_context = set()
            
            for scene in scenes:
                # Handle objects
                for obj in scene.get('objects', []):
                    if isinstance(obj, dict):
                        objects_context.add(obj.get('name', ''))
                    else:
                        objects_context.add(str(obj))
                
                # Handle scenes
                for scene_item in scene.get('scenes', []):
                    if isinstance(scene_item, dict):
                        scenes_context.add(scene_item.get('scene', ''))
                    else:
                        scenes_context.add(str(scene_item))
            
            # Remove empty strings and limit context size
            objects_context = list(filter(None, objects_context))[:10]
            scenes_context = list(filter(None, scenes_context))[:5]
            
            # Create context
            context = f"""
            Objects detected: {', '.join(objects_context) if objects_context else 'none'}
            Scene types: {', '.join(scenes_context) if scenes_context else 'none'}
            """
            
            story_prompt = f"Write a short story about: {prompt}. {context}"
            
            inputs = self.tokenizer(story_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.story_model.generate(
                    inputs["input_ids"],
                    max_new_tokens=200,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    do_sample=True
                )
            
            story = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            sentences = [s.strip() for s in story.split('.') if s.strip()]
            return sentences[:5]
            
        except Exception as e:
            logger.error(f"Story generation error: {str(e)}")
            return ["Once upon a time...", 
                    "Something interesting happened.", 
                    "The end."]

    def match_clips(self, story_sentences: List[str], scenes: List[Dict], frame_timestamps: List[float]) -> List[Dict]:
        """Match story sentences to video clips using CLIP"""
        try:
            matched_clips = []
            
            for sentence in story_sentences:
                # Encode sentence with CLIP
                with torch.no_grad():
                    text = clip.tokenize([sentence]).to(self.device)
                    text_embedding = self.clip_model.encode_text(text)
                
                # Search in FAISS index
                D, I = self.faiss_index.search(text_embedding.cpu().numpy(), 1)
                best_match_idx = I[0][0]
                
                # Get corresponding timestamp
                timestamp = frame_timestamps[best_match_idx]
                
                matched_clips.append({
                    'start': max(0, timestamp - 1),  # Start 1 second before
                    'duration': 4.0,  # 4-second clips
                    'sentence': sentence
                })
            
            return matched_clips
            
        except Exception as e:
            logger.error(f"Clip matching error: {str(e)}")
            return []

    def generate_music(self, prompt: str) -> str:
        """Generate background music"""
        try:
            # Simple background audio using FFmpeg
            music_filename = f'music_{int(time.time())}.wav'
            music_path = os.path.join(TEMP_FOLDER, music_filename)
            
            command = [
                'ffmpeg', '-y',
                '-f', 'lavfi',
                '-i', 'anoisesrc=c=pink:d=30',  # 30 seconds of pink noise
                '-af', 'volume=0.2',  # Lower volume
                music_path
            ]
            
            subprocess.run(command, check=True, capture_output=True)
            return music_path if os.path.exists(music_path) else None
            
        except Exception as e:
            logger.error(f"Music generation error: {str(e)}")
            return None

    async def generate(self, video_path: str, prompt: str):
        """Main generation pipeline"""
        clips = []
        music_path = None
        try:
            output_filename = f'output_{int(time.time())}.mp4'
            final_output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            # Scene Detection
            yield json.dumps({
                'status': 'processing',
                'message': 'Analyzing video content...',
                'progress': 20
            }) + "\n"
            
            scenes = self.detect_scenes(video_path)
            
            # Story Generation
            yield json.dumps({
                'status': 'processing',
                'message': 'Generating story...',
                'progress': 40
            }) + "\n"
            
            story_sentences = self.generate_story(prompt, scenes)
            
            # Clip Matching
            yield json.dumps({
                'status': 'processing',
                'message': 'Matching clips to story...',
                'progress': 60
            }) + "\n"
            
            matched_clips = self.match_clips(story_sentences, scenes, [scene['timestamp'] for scene in scenes])
            
            # Extract Clips
            clips = self.extract_clips(video_path, scenes)
            
            # Generate Music
            yield json.dumps({
                'status': 'processing',
                'message': 'Creating background music...',
                'progress': 80
            }) + "\n"
            
            music_path = self.generate_music(prompt)
            
            # Final Combination
            yield json.dumps({
                'status': 'processing',
                'message': 'Creating final video...',
                'progress': 90
            }) + "\n"
            
            success = self.combine_clips(clips, final_output_path, TEMP_FOLDER)
            
            if success:
                yield json.dumps({
                    'status': 'complete',
                    'message': 'Video story created successfully!',
                    'output_path': output_filename,
                    'story': story_sentences
                }) + "\n"
            else:
                raise Exception("Failed to create final video")

        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            yield json.dumps({
                'status': 'error',
                'message': f'Error creating video story: {str(e)}'
            }) + "\n"

        finally:
            # Cleanup
            for clip in clips:
                if os.path.exists(clip):
                    os.remove(clip)
            if music_path and os.path.exists(music_path):
                os.remove(music_path)

    def _extract_frames(self, video_path: str):
        """Extract frames from video for processing"""
        # TODO: Implement frame extraction
        pass

    def _process_frames(self, frames, prompt: str):
        """Process video frames with CLIP model"""
        # TODO: Implement CLIP processing
        pass

    def add_cinematic_effects(self, clip):
        """Add cinematic effects to a clip"""
        try:
            # Add slight color grading
            clip = clip.fx(colorx, 1.2)  # Color enhancement
            
            # Add subtle vignette
            clip = clip.fx(vfx.vfx_vignette, 0.85)  # Vignette
            
            # Add slight zoom effect
            clip = clip.fx(zoom, 1.03)  # Subtle zoom
            
            # Stabilize brightness
            clip = clip.fx(vfx.lum_contrast)  # Contrast
            
            return clip
        except Exception as e:
            logger.warning(f"Error adding effects: {str(e)}")
            return clip

    def create_transition(self, clip1, clip2, transition_duration=1.0):
        """Create transition between clips"""
        try:
            # Crossfade transition
            return CompositeVideoClip([
                clip1.crossfadeout(transition_duration),
                clip2.crossfadein(transition_duration)
            ])
        except Exception as e:
            logger.warning(f"Error creating transition: {str(e)}")
            return clip1

    def combine_clips(self, clips: List[str], output_path: str, temp_dir: str) -> bool:
        """Combine video clips using MoviePy"""
        try:
            if not clips:
                logger.error("No clips to combine")
                return False
            
            video_clips = []
            
            for i, clip_path in enumerate(clips):
                try:
                    if not os.path.exists(clip_path):
                        logger.warning(f"Clip not found: {clip_path}")
                        continue
                        
                    # Load video clip with error handling
                    try:
                        video = VideoFileClip(clip_path, audio=True)
                        
                        # Verify the clip loaded correctly
                        if video.reader is None or video.duration <= 0:
                            logger.warning(f"Failed to load clip properly: {clip_path}")
                            continue
                            
                        # Ensure the clip is at least 1 second long
                        if video.duration < 1:
                            logger.warning(f"Clip too short: {clip_path}")
                            continue
                            
                        # Trim clip to avoid end-of-file issues
                        safe_duration = max(1, min(video.duration - 0.1, 4.0))
                        video = video.subclip(0, safe_duration)
                        
                        video_clips.append(video)
                        logger.info(f"Loaded clip {i+1}/{len(clips)} - Duration: {safe_duration:.2f}s")
                        
                    except Exception as e:
                        logger.error(f"Error loading clip {clip_path}: {str(e)}")
                        continue
                        
                except Exception as e:
                    logger.error(f"Error processing clip {clip_path}: {str(e)}")
                    continue
            
            if not video_clips:
                logger.error("No clips were successfully loaded")
                return False
            
            try:
                # Combine all clips with simple concatenation
                final_clip = concatenate_videoclips(
                    video_clips,
                    method="compose",
                    padding=0  # No padding between clips
                )
                
                # Write final video with safe settings
                final_clip.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile=os.path.join(temp_dir, 'temp-audio.m4a'),
                    remove_temp=True,
                    fps=30,
                    preset='ultrafast',
                    threads=4,
                    ffmpeg_params=['-strict', '-2']
                )
                
                # Close all clips
                final_clip.close()
                for clip in video_clips:
                    clip.close()
                
                if os.path.exists(output_path):
                    logger.info(f"Successfully combined {len(video_clips)} clips into: {output_path}")
                    return True
                else:
                    logger.error("Output file was not created")
                    return False
                
            except Exception as e:
                logger.error(f"Error combining clips: {str(e)}")
                return False
            
        except Exception as e:
            logger.error(f"Error in combine_clips: {str(e)}")
            return False
        finally:
            # Cleanup
            try:
                for clip in video_clips:
                    try:
                        clip.close()
                    except:
                        pass
            except Exception as e:
                logger.warning(f"Cleanup error: {str(e)}")

    def extract_clips(self, video_path: str, scenes: List[Dict], output_dir: str = None) -> List[str]:
        """Extract clips using MoviePy"""
        try:
            if not scenes:
                logger.warning(f"No scenes to extract from {video_path}")
                return []
            
            if output_dir is None:
                output_dir = self.clips_dir
            
            extracted_clips = []
            
            # Load the video
            with VideoFileClip(video_path) as video:
                for scene in scenes:
                    if not isinstance(scene, dict):
                        continue
                        
                    timestamp = scene.get('timestamp', 0)
                    if not isinstance(timestamp, (int, float)):
                        continue
                    
                    try:
                        # Create subclip
                        clip = video.subclip(timestamp, None)  # Extract from timestamp to end
                        
                        # Generate output path
                        video_id = os.path.splitext(os.path.basename(video_path))[0]
                        clip_filename = f'clip_{video_id}_{int(timestamp)}_{int(time.time())}.mp4'
                        clip_path = os.path.join(output_dir, clip_filename)
                        
                        # Write clip
                        clip.write_videofile(
                            clip_path,
                            codec='libx264',
                            audio_codec='aac',
                            preset='ultrafast',
                            fps=30,
                            threads=4,
                            logger=None
                        )
                        
                        if os.path.exists(clip_path):
                            extracted_clips.append(clip_path)
                            logger.info(f"Extracted clip at {timestamp:.2f}s from {video_path}")
                        
                    except Exception as e:
                        logger.error(f"Error extracting clip at {timestamp:.2f}s: {str(e)}")
                        continue
                        
            if not extracted_clips:
                logger.warning("No clips were successfully extracted")
            else:
                logger.info(f"Successfully extracted {len(extracted_clips)} clips")
            
            return extracted_clips
            
        except Exception as e:
            logger.error(f"Clip extraction error: {str(e)}")
            return []

    def generate_thumbnail(self, video_path: str) -> str:
        """Generate thumbnail from video"""
        try:
            # Generate thumbnail filename
            thumbnail_name = f"thumb_{int(time.time())}_{os.path.basename(video_path)}.jpg"
            thumbnail_path = os.path.join(self.output_dir, thumbnail_name)

            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Could not open video file")

            # Get total frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                raise Exception("Video has no frames")

            # Set position to middle frame
            middle_frame = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)

            # Read frame
            ret, frame = cap.read()
            if not ret:
                raise Exception("Could not read frame")

            # Resize if needed (optional)
            max_size = 500
            height, width = frame.shape[:2]
            if height > max_size or width > max_size:
                scale = max_size / max(height, width)
                frame = cv2.resize(frame, None, fx=scale, fy=scale)

            # Save thumbnail
            cv2.imwrite(thumbnail_path, frame)
            cap.release()

            logger.info(f"Generated thumbnail: {thumbnail_name}")
            return thumbnail_name

        except Exception as e:
            logger.error(f"Error generating thumbnail: {str(e)}")
            return None

    def process_video(self, video_path: str) -> Dict:
        """Process a single video and generate thumbnail"""
        try:
            thumbnail = self.generate_thumbnail(video_path)
            return {
                'path': video_path,
                'thumbnail': thumbnail,
                'filename': os.path.basename(video_path)
            }
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return None

    def create_final_video(self, video_paths: List[str], prompt: str) -> Tuple[str, str, str]:
        """Create final video with thumbnails"""
        try:
            # Process each video
            processed_videos = []
            for video_path in video_paths:
                result = self.process_video(video_path)
                if result:
                    processed_videos.append(result)

            # Generate output path
            timestamp = int(time.time())
            output_filename = f"final_{timestamp}.mp4"
            output_path = os.path.join(self.output_dir, output_filename)

            # Combine videos
            success = self.combine_clips(video_paths, output_path, self.temp_dir)
            
            if success:
                # Generate thumbnail for final video
                final_thumbnail = self.generate_thumbnail(output_path)
                
                return output_path, final_thumbnail, "Your story has been generated!"
            
            return None, None, None

        except Exception as e:
            logger.error(f"Error in create_final_video: {str(e)}")
            return None, None, None

if __name__ == "__main__":
    logger.info("Video service initialized")