from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import os
import aiofiles
import logging
import time
import shutil
import uvicorn
from pathlib import Path
from werkzeug.utils import secure_filename
from fastapi.staticfiles import StaticFiles

from video_service import VideoUnderstandingService, UPLOAD_FOLDER, OUTPUT_FOLDER

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Video Processing Service",
    description="API for video uploads and processing",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize video service
video_service = VideoUnderstandingService()

# Mount the output directory
app.mount("/videos", StaticFiles(directory=OUTPUT_FOLDER), name="videos")

@app.post("/upload")
async def upload_video(files: List[UploadFile] = File(...), prompt: str = Form(...)):
    try:
        video_paths = []
        for file in files:
            temp_file = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}_{file.filename}")
            with open(temp_file, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            video_paths.append(temp_file)

        output_path, thumbnail, story = video_service.create_final_video(video_paths, prompt)

        if output_path and os.path.exists(output_path):
            video_filename = os.path.basename(output_path)
            video_url = f"http://localhost:8001/videos/{video_filename}"
            
            response_data = {
                "video": video_filename,
                "videoUrl": video_url,
                "story": story
            }
            
            # Add thumbnail if available
            if thumbnail:
                response_data["thumbnailUrl"] = f"http://localhost:8001/videos/{thumbnail}"
                logger.info(f"Thumbnail URL: {response_data['thumbnailUrl']}")

            return response_data
        else:
            raise HTTPException(status_code=500, detail="Video processing failed")

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup uploaded files
        for path in video_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                logger.warning(f"Cleanup error: {str(e)}")

@app.post("/process/{filename}")
async def process_video(filename: str, prompt: str = Form(...)):
    """Process an already uploaded video"""
    try:
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Video not found")
            
        return StreamingResponse(
            video_service.generate(video_path, prompt),
            media_type="application/x-ndjson"
        )
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video/{video_name}")
async def get_video(video_name: str):
    video_path = os.path.join(OUTPUT_FOLDER, video_name)
    if not os.path.exists(video_path):
        return {"error": "Video not found"}
    return FileResponse(video_path, media_type="video/mp4")

@app.get("/uploads/{filename}")
async def get_upload(filename: str):
    """Serve uploaded files"""
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@app.delete("/uploads/{filename}")
async def delete_upload(filename: str):
    """Delete uploaded file"""
    try:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return JSONResponse({
                "status": "success",
                "message": f"Deleted file {filename}"
            })
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    try:
        logger.info("Starting server on port 8001...")
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8001,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Server startup failed: {str(e)}")
        raise
