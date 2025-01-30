const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const fs = require('fs');

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// Configure multer for video upload
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadDir = 'uploads/';
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir);
    }
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + '-' + file.originalname);
  }
});

const upload = multer({ 
  storage: storage,
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('video/')) {
      cb(null, true);
    } else {
      cb(new Error('Not a video file!'), false);
    }
  },
  limits: {
    fileSize: 100 * 1024 * 1024, // 100MB limit
  }
});

// Serve uploaded files statically
app.use('/uploads', express.static('uploads'));

// Routes
app.post('/generate', upload.array('videos'), async (req, res) => {
  try {
    const videoFiles = req.files;
    const prompt = req.body.prompt;

    if (!videoFiles || videoFiles.length === 0) {
      return res.status(400).json({ error: 'No video files uploaded' });
    }

    // Process videos and generate story
    const processedVideo = await processVideos(videoFiles, prompt);
    
    // Generate background music
    const backgroundMusic = await generateBackgroundMusic(prompt);

    res.json({
      videoUrl: processedVideo.url,
      backgroundMusic: backgroundMusic.url
    });

  } catch (error) {
    console.error('Error processing request:', error);
    res.status(500).json({ 
      error: 'Error processing your request',
      details: error.message 
    });
  }
});

// Helper functions for video processing and music generation
async function processVideos(files, prompt) {
  // TODO: Implement actual video processing logic
  // This is a placeholder that returns a mock response
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        url: `/uploads/${files[0].filename}`,
        duration: 60
      });
    }, 2000);
  });
}

async function generateBackgroundMusic(prompt) {
  // TODO: Implement actual music generation logic
  // This is a placeholder that returns a mock response
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        url: '/uploads/background-music.mp3',
        duration: 60
      });
    }, 1500);
  });
}

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ 
    error: 'Something broke!',
    details: err.message 
  });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
}); 