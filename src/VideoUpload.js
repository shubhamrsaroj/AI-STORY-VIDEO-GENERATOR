import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  Box,
  Button,
  Container,
  Typography,
  TextField,
  Stack,
  Paper,
  LinearProgress,
  Card,
  CardContent,
  IconButton,
  Chip,
  Alert,
  Fade,
  CircularProgress,
  Grid,
  Tabs,
  Tab,
  Divider,
  CardMedia,
  CardActions,
  Grow,
  useTheme
} from '@mui/material';
import {
  CloudUpload as CloudUploadIcon,
  Close as CloseIcon,
  Movie as MovieIcon,
  Search as SearchIcon,
  AutoStories as StoryIcon,
  Description,
  Delete as DeleteIcon,
  PlayArrow as PlayArrowIcon,
  AccessTime as AccessTimeIcon,
  VideoLibrary as VideoLibraryIcon
} from '@mui/icons-material';
import { styled, keyframes } from '@mui/material/styles';
import ImageList from '@mui/material/ImageList';
import ImageListItem from '@mui/material/ImageListItem';
import ImageListItemBar from '@mui/material/ImageListItemBar';
import { PlayCircleOutline, Image } from '@mui/icons-material';
import { alpha } from '@mui/material/styles';
import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';

// API configuration
const API_BASE_URL = 'http://localhost:8001';
const API_ENDPOINTS = {
  UPLOAD: '/upload',
  GENERATE: '/generate',
  SEARCH: '/search',
  OUTPUT: '/output'
};

// File upload configuration
const MAX_FILE_SIZE = 1024 * 1024 * 1024; // 1GB
const SUPPORTED_TYPES = {
  video: ['video/mp4', 'video/mov', 'video/avi', 'video/mkv', 'video/webm'],
  image: ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
};

// Animated gradient background with dark theme
const gradientAnimation = keyframes`
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
`;

const StyledContainer = styled(Container)(({ theme }) => ({
  minHeight: '100vh',
  background: 'linear-gradient(135deg, #0A0A0A 0%, #1A1A1A 100%)',
  padding: theme.spacing(4),
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing(4)
}));

const UploadZone = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(4),
  textAlign: 'center',
  background: 'rgba(30, 30, 30, 0.6)',
  backdropFilter: 'blur(10px)',
  borderRadius: theme.spacing(2),
  border: '2px dashed rgba(100, 100, 100, 0.3)',
  transition: 'all 0.3s ease',
  '&:hover': {
    border: '2px dashed rgba(0, 255, 157, 0.5)',
    transform: 'translateY(-4px)',
    boxShadow: '0 8px 20px rgba(0, 255, 157, 0.2)'
  }
}));

const VideoCard = styled(Card)(({ theme }) => ({
  height: '100%',
  background: 'rgba(30, 30, 30, 0.6)',
  backdropFilter: 'blur(10px)',
  borderRadius: theme.spacing(2),
  overflow: 'hidden',
  transition: 'all 0.3s ease',
  border: '1px solid rgba(50, 50, 50, 0.5)',
  '&:hover': {
    transform: 'translateY(-4px)',
    boxShadow: '0 8px 20px rgba(0, 255, 157, 0.2)',
    border: '1px solid rgba(0, 255, 157, 0.3)'
  }
}));

const StyledButton = styled(Button)(({ theme }) => ({
  borderRadius: theme.spacing(3),
  padding: theme.spacing(1.5, 4),
  textTransform: 'none',
  fontSize: '1.1rem',
  fontWeight: 600,
  background: 'linear-gradient(45deg, #00ff9d 30%, #00f2fe 90%)',
  color: '#0A0A0A',
  '&:hover': {
    background: 'linear-gradient(45deg, #00f2fe 30%, #00ff9d 90%)',
  },
  '&.Mui-disabled': {
    background: 'rgba(50, 50, 50, 0.5)',
    color: 'rgba(255, 255, 255, 0.3)'
  }
}));

const StyledImageList = styled(ImageList)(({ theme }) => ({
  width: '100%',
  height: 450,
  transform: 'translateZ(0)',
  '& .MuiImageListItem-root': {
    overflow: 'hidden',
    borderRadius: theme.shape.borderRadius,
    transition: 'all 0.3s ease',
    '&:hover': {
      transform: 'scale(1.02)',
      boxShadow: '0 4px 20px rgba(0, 0, 0, 0.12)',
    }
  }
}));

const PreviewCard = styled(Card)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  position: 'relative',
  overflow: 'hidden',
  transition: 'all 0.3s ease',
  background: alpha(theme.palette.background.paper, 0.8),
  backdropFilter: 'blur(10px)',
  '&:hover': {
    transform: 'translateY(-4px) scale(1.02)',
    boxShadow: theme.shadows[8],
    '& .preview-overlay': {
      opacity: 1,
    }
  }
}));

const FileTypeChip = styled(Chip)(({ theme, filetype }) => ({
  position: 'absolute',
  top: 8,
  right: 8,
  backgroundColor: filetype === 'video' 
    ? alpha(theme.palette.error.main, 0.9)
    : alpha(theme.palette.success.main, 0.9),
  color: theme.palette.common.white,
  backdropFilter: 'blur(4px)',
}));

const VisuallyHiddenInput = styled('input')`
  clip: rect(0 0 0 0);
  clip-path: inset(50%);
  height: 1px;
  overflow: hidden;
  position: absolute;
  bottom: 0;
  left: 0;
  white-space: nowrap;
  width: 1px;
`;

const MediaCard = styled(Card)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  position: 'relative',
  '&:hover .media-overlay': {
    opacity: 1,
  },
  '&:hover .play-button': {
    opacity: 1,
  },
}));

const MediaOverlay = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  backgroundColor: 'rgba(0, 0, 0, 0.3)',
  opacity: 0,
  transition: 'opacity 0.3s ease',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
}));

const PlayButton = styled(IconButton)(({ theme }) => ({
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  backgroundColor: 'rgba(0, 255, 157, 0.2)',
  '&:hover': {
    backgroundColor: 'rgba(0, 255, 157, 0.4)',
  },
  '& .MuiSvgIcon-root': {
    color: '#00ff9d',
    fontSize: 40
  }
}));

const StyledTextField = styled(TextField)(({ theme }) => ({
  '& .MuiOutlinedInput-root': {
    color: '#fff',
    backgroundColor: 'rgba(30, 30, 30, 0.6)',
    '& fieldset': {
      borderColor: 'rgba(100, 100, 100, 0.3)',
    },
    '&:hover fieldset': {
      borderColor: 'rgba(0, 255, 157, 0.5)',
    },
    '&.Mui-focused fieldset': {
      borderColor: '#00ff9d',
    },
  },
  '& .MuiInputLabel-root': {
    color: 'rgba(255, 255, 255, 0.7)',
  },
}));

const GenerateButton = styled(motion.button)(({ theme }) => ({
  width: '100%',
  padding: '12px 32px',
  fontSize: '1.1rem',
  border: 'none',
  borderRadius: 30,
  color: '#1a1a1a',
  cursor: 'pointer',
  background: 'linear-gradient(45deg, #00ff9d 30%, #00f2fe 90%)',
  boxShadow: '0 3px 5px 2px rgba(0, 255, 157, .3)',
  '&:hover': {
    background: 'linear-gradient(45deg, #00f2fe 30%, #00ff9d 90%)',
  },
  '&:disabled': {
    opacity: 0.7,
    cursor: 'not-allowed',
  },
}));

const ProgressBar = styled(LinearProgress)(({ theme }) => ({
  height: 10,
  borderRadius: 5,
  backgroundColor: 'rgba(255, 255, 255, 0.1)',
  '& .MuiLinearProgress-bar': {
    borderRadius: 5,
    background: 'linear-gradient(45deg, #00ff9d 30%, #00f2fe 90%)',
  },
}));

const StyledAlert = styled(Alert)(({ theme, severity }) => ({
  backgroundColor: severity === 'error' 
    ? 'rgba(255, 50, 50, 0.1)' 
    : 'rgba(0, 255, 157, 0.1)',
  color: severity === 'error' ? '#ff5050' : '#00ff9d',
  border: `1px solid ${severity === 'error' ? '#ff5050' : '#00ff9d'}`,
  borderRadius: 12,
  '& .MuiAlert-icon': {
    color: severity === 'error' ? '#ff5050' : '#00ff9d',
  },
}));

const PromptSection = styled(Paper)(({ theme }) => ({
    backgroundColor: 'rgba(40, 44, 52, 0.9)',
    padding: theme.spacing(3),
    borderRadius: 16,
    border: '1px solid rgba(255, 255, 255, 0.1)',
    marginTop: theme.spacing(4),
    marginBottom: theme.spacing(4),
}));

const SearchField = styled(TextField)(({ theme }) => ({
    '& .MuiOutlinedInput-root': {
        color: '#fff',
        backgroundColor: 'rgba(40, 44, 52, 0.9)',
        '& fieldset': {
            borderColor: 'rgba(255, 255, 255, 0.2)',
        },
        '&:hover fieldset': {
            borderColor: '#00ff9d',
        },
        '&.Mui-focused fieldset': {
            borderColor: '#00ff9d',
        },
    },
    '& .MuiInputLabel-root': {
        color: 'rgba(255, 255, 255, 0.7)',
    },
}));

const VideoThumbnail = styled(Card)(({ theme }) => ({
    backgroundColor: 'rgba(40, 44, 52, 0.8)',
    borderRadius: 16,
    overflow: 'hidden',
    transition: 'transform 0.3s ease',
    '&:hover': {
        transform: 'translateY(-5px)',
        '& .play-overlay': {
            opacity: 1
        }
    }
}));

const ThumbnailOverlay = styled(Box)({
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'rgba(0, 0, 0, 0.4)',
    opacity: 0,
    transition: 'opacity 0.3s ease'
});

const Duration = styled(Box)({
    position: 'absolute',
    bottom: 8,
    right: 8,
    padding: '4px 8px',
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    color: '#fff',
    borderRadius: 4,
    fontSize: '0.875rem'
});

const VideoTitle = styled(Typography)(({ theme }) => ({
    color: '#fff',
    fontSize: '14px',
    fontWeight: 'bold',
    marginTop: theme.spacing(1),
    lineHeight: 1.2,
    display: '-webkit-box',
    WebkitLineClamp: 2,
    WebkitBoxOrient: 'vertical',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
}));

const VideoInfo = styled(Typography)(({ theme }) => ({
    color: 'rgba(255, 255, 255, 0.7)',
    fontSize: '12px',
    marginTop: theme.spacing(0.5),
}));

const SearchBar = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(2),
  width: '100%',
  marginBottom: theme.spacing(4)
}));

const SearchTextField = styled(TextField)(({ theme }) => ({
  flex: 1,
  '& .MuiOutlinedInput-root': {
    color: '#fff',
    backgroundColor: 'rgba(30, 30, 30, 0.6)',
    '& fieldset': {
      borderColor: 'rgba(100, 100, 100, 0.3)',
    },
    '&:hover fieldset': {
      borderColor: 'rgba(0, 255, 157, 0.5)',
    },
    '&.Mui-focused fieldset': {
      borderColor: '#00ff9d',
    },
  },
  '& .MuiInputLabel-root': {
    color: 'rgba(255, 255, 255, 0.7)',
  },
}));

const VideoUpload = () => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [fileUrls, setFileUrls] = useState([]);
  const [prompt, setPrompt] = useState('');
  const [duration, setDuration] = useState(30);
  const [loading, setLoading] = useState(false);
  const [videoUrl, setVideoUrl] = useState('');
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState(0);
  const [searchQuery, setSearchQuery] = useState('');
  const [transcript, setTranscript] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const fileInputRef = useRef(null);
  const [fileError, setFileError] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [success, setSuccess] = useState(null);
  const [generatedVideo, setGeneratedVideo] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isSearching, setIsSearching] = useState(false);
  const [thumbnail, setThumbnail] = useState('');
  const [result, setResult] = useState(null);
  const [uploadedVideos, setUploadedVideos] = useState([]);
  
  const theme = useTheme();
  
  // Animation variants for Framer Motion
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1
    }
  };

  // Cleanup URLs when component unmounts or files change
  useEffect(() => {
    return () => {
      fileUrls.forEach(url => URL.revokeObjectURL(url));
    };
  }, [fileUrls]);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const files = Array.from(event.dataTransfer.files);
    handleFileSelect({ target: { files } });
  };

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files);
    const validFiles = files.filter(file => {
        const isValid = file.type.startsWith('video/');
        const isValidSize = file.size <= MAX_FILE_SIZE;
        
        if (!isValid) {
            setError('Please upload only video files');
            return false;
        }
        if (!isValidSize) {
            setError(`File ${file.name} is too large. Maximum size is 1GB`);
            return false;
        }
        return true;
    });

    if (validFiles.length > 0) {
        // Create URLs for new files
        const newUrls = validFiles.map(file => URL.createObjectURL(file));
        
        setSelectedFiles(prev => [...prev, ...validFiles]);
        setFileUrls(prev => [...prev, ...newUrls]);
        setError(null);
    }
  };

  const handleRemoveFile = (index) => {
    // Revoke URL for removed file
    URL.revokeObjectURL(fileUrls[index]);
    
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
    setFileUrls(prev => prev.filter((_, i) => i !== index));
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    setIsSearching(true);
    try {
        const response = await fetch(`http://localhost:8001/search?query=${encodeURIComponent(searchQuery)}`);
        if (!response.ok) throw new Error('Search failed');
        
        const data = await response.json();
        setSearchResults(data);
    } catch (error) {
        console.error('Search error:', error);
    } finally {
        setIsSearching(false);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const onDrop = useCallback((acceptedFiles) => {
    const validFiles = acceptedFiles.filter(file => 
        SUPPORTED_TYPES.video.includes(file.type) && file.size <= MAX_FILE_SIZE
    );

    if (validFiles.length === 0) {
        setError('Please upload a valid video file (MP4, MOV, AVI, or MKV, max 1GB)');
        return;
    }

    setSelectedFiles(validFiles);
    setError(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
        'video/*': SUPPORTED_TYPES.video
    },
    maxSize: MAX_FILE_SIZE,
    multiple: true
  });

  const handleGenerate = async (e) => {
    e.preventDefault();
    
    if (selectedFiles.length === 0) {
        setError('Please select at least one video file');
        return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);
    setUploadProgress(0);

    try {
        const formData = new FormData();
        
        // Add all selected files
        selectedFiles.forEach((file, index) => {
            formData.append('files', file);
        });
        
        // Add the prompt
        formData.append('prompt', prompt.trim());

        const response = await axios.post(
            `${API_BASE_URL}/upload`,
            formData,
            {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                onUploadProgress: (progressEvent) => {
                    const progress = (progressEvent.loaded / progressEvent.total) * 100;
                    setUploadProgress(Math.round(progress));
                }
            }
        );

        // Handle streaming response
        const lines = response.data.split('\n').filter(line => line.trim());
        for (const line of lines) {
            try {
                const data = JSON.parse(line);
                
                if (data.status === 'error') {
                    setError(data.message);
                    break;
                }
                
                if (data.status === 'processing') {
                    setUploadProgress(data.progress || 0);
                }
                
                if (data.status === 'complete' && data.output_path) {
                    setGeneratedVideo(`${API_BASE_URL}/video/${data.output_path}`);
                    setSuccess('Video story generated successfully!');
                    setResult({
                        story: data.story,
                        videoUrl: data.videoUrl,
                        thumbnailUrl: data.thumbnailUrl
                    });
                }
            } catch (err) {
                console.error('Error parsing response:', err);
            }
        }

    } catch (err) {
        console.error('Generation error:', err);
        setError(err.response?.data?.detail || 'Error generating video');
    } finally {
        setLoading(false);
    }
  };

  const getFilePreview = (file) => {
    if (file.type.startsWith('video/')) {
      return URL.createObjectURL(file);
    }
    return file.type.startsWith('image/') ? URL.createObjectURL(file) : null;
  };

  const formatDuration = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const renderMediaGrid = () => (
    <Grid container spacing={2} sx={{ mt: 4 }}>
        {selectedFiles.map((file, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
                <VideoThumbnail>
                    <Box sx={{ position: 'relative', paddingTop: '56.25%' }}>
                        <video
                            src={fileUrls[index]}
                            style={{
                                position: 'absolute',
                                top: 0,
                                left: 0,
                                width: '100%',
                                height: '100%',
                                objectFit: 'cover',
                            }}
                            onLoadedMetadata={(e) => {
                                const video = e.target;
                                // Store video duration
                                const newFiles = [...selectedFiles];
                                newFiles[index] = {
                                    ...newFiles[index],
                                    duration: video.duration
                                };
                                setSelectedFiles(newFiles);
                                
                                // Generate thumbnail
                                const canvas = document.createElement('canvas');
                                canvas.width = video.videoWidth;
                                canvas.height = video.videoHeight;
                                canvas.getContext('2d').drawImage(video, 0, 0);
                            }}
                        />
                        <ThumbnailOverlay>
                            <PlayButton>
                                <PlayArrowIcon sx={{ color: '#fff', fontSize: 30 }} />
                            </PlayButton>
                        </ThumbnailOverlay>
                        <Duration>
                            {formatDuration(file.duration || 0)}
                        </Duration>
                    </Box>
                    <CardContent sx={{ p: 2 }}>
                        <Typography variant="subtitle1" sx={{ color: '#fff', mb: 1 }}>
                            {file.name}
                        </Typography>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                                {(file.size / (1024 * 1024)).toFixed(2)} MB
                            </Typography>
                            <IconButton 
                                onClick={() => handleRemoveFile(index)}
                                sx={{ color: '#ff4d4d' }}
                            >
                                <DeleteIcon />
                            </IconButton>
                        </Box>
                    </CardContent>
                </VideoThumbnail>
              
            </Grid>
        ))}
         
    </Grid>
     
  );

  // Error display component
  const ErrorDisplay = ({ error }) => {
    if (!error) return null;
    
    // If error is an object, convert it to string
    const errorMessage = typeof error === 'object' ? JSON.stringify(error) : error;
    
    return (
      <StyledAlert severity="error" sx={{ mt: 2 }}>
        {errorMessage}
      </StyledAlert>
    );
  };

  const generateThumbnail = async (file) => {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const video = document.createElement('video');
            video.preload = 'metadata';

            video.onloadedmetadata = () => {
                video.currentTime = 1; // Get frame from 1 second in
            };

            video.onseeked = () => {
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                resolve(canvas.toDataURL());
            };

            video.src = e.target.result;
        };
        reader.readAsDataURL(file);
    });
  };

  const handleFileChange = async (event) => {
    try {
        const selectedFiles = Array.from(event.target.files);
        setSelectedFiles(selectedFiles);
        
        const videoData = await Promise.all(
            selectedFiles.map(async (file) => {
                const thumbnail = await generateThumbnail(file);
                return {
                    name: file.name,
                    url: URL.createObjectURL(file),
                    thumbnail: thumbnail,
                    size: file.size
                };
            })
        );
        
        console.log('Video data:', videoData); // Debug log
        setUploadedVideos(videoData);
        setError(null);
    } catch (err) {
        console.error('Error handling files:', err);
        setError('Error processing videos');
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFiles.length || !prompt) return;

    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('files', file);
    });
    formData.append('prompt', prompt);

    try {
        const response = await fetch('http://localhost:8001/upload', {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            throw new Error('Upload failed');
        }
        
        const data = await response.json();
        console.log('Response data:', data); // Debug log
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        setResult(data);
    } catch (err) {
        console.error('Upload error:', err);
        setError(err.message);
    } finally {
        setLoading(false);
    }
  };

  // Cleanup URLs on unmount
  useEffect(() => {
    return () => {
        uploadedVideos.forEach(video => {
            if (video.url) {
                URL.revokeObjectURL(video.url);
            }
        });
    };
  }, [uploadedVideos]);

  return (
    <StyledContainer maxWidth="xl">
      <Typography 
        variant="h3" 
        align="center" 
        sx={{ 
          color: '#00ff9d',
          mb: 4,
          fontWeight: 'bold',
          textShadow: '0 0 10px rgba(0, 255, 157, 0.3)'
        }}
      >
        Video Story Generator
      </Typography>

      {/* Add Search Section */}
      <SearchBar>
        <SearchTextField
          fullWidth
          label="Search videos"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
          InputProps={{
            endAdornment: (
              <IconButton 
                onClick={handleSearch}
                disabled={isSearching}
                sx={{ 
                  color: '#00ff9d',
                  '&:hover': {
                    backgroundColor: 'rgba(0, 255, 157, 0.1)'
                  }
                }}
              >
                {isSearching ? <CircularProgress size={24} sx={{ color: '#00ff9d' }} /> : <SearchIcon />}
              </IconButton>
            ),
          }}
        />
      </SearchBar>

      {/* Search Results */}
      {searchResults.length > 0 && (
        <Fade in={true}>
          <Box sx={{ mb: 4 }}>
            <Typography 
              variant="h5" 
              sx={{ 
                color: '#00ff9d',
                mb: 3,
                display: 'flex',
                alignItems: 'center',
                gap: 1
              }}
            >
              <SearchIcon /> Search Results
            </Typography>
            <Grid container spacing={3}>
              {searchResults.map((video, index) => (
                <Grid item xs={12} sm={6} md={4} key={`search-${index}`}>
                  <VideoCard>
                    <Box sx={{ position: 'relative' }}>
                      <Box sx={{ 
                        pt: '56.25%', 
                        position: 'relative',
                        bgcolor: 'black' 
                      }}>
                        {video.thumbnailUrl && (
                          <CardMedia
                            component="img"
                            image={video.thumbnailUrl}
                            alt={video.title}
                            sx={{
                              position: 'absolute',
                              top: 0,
                              left: 0,
                              width: '100%',
                              height: '100%',
                              objectFit: 'contain'
                            }}
                          />
                        )}
                        <PlayButton onClick={() => window.open(video.videoUrl, '_blank')}>
                          <PlayArrowIcon />
                        </PlayButton>
                      </Box>
                    </Box>
                    <CardContent>
                      <Typography variant="subtitle1" sx={{ color: 'white' }}>
                        {video.title}
                      </Typography>
                      {video.description && (
                        <Typography 
                          variant="body2" 
                          sx={{ 
                            color: 'rgba(255, 255, 255, 0.7)',
                            mt: 1 
                          }}
                        >
                          {video.description}
                        </Typography>
                      )}
                    </CardContent>
                  </VideoCard>
                </Grid>
              ))}
            </Grid>
          </Box>
        </Fade>
      )}

      {/* Upload Section */}
      <UploadZone elevation={4}>
        <form onSubmit={handleSubmit}>
          <Grid container spacing={4}>
            <Grid item xs={12}>
              <StyledButton
                component="label"
                startIcon={<CloudUploadIcon />}
                fullWidth
                sx={{ height: '120px', fontSize: '1.2rem' }}
              >
                Drop your videos here or click to upload
                <input
                  type="file"
                  hidden
                  multiple
                  accept="video/*"
                  onChange={handleFileChange}
                />
              </StyledButton>
            </Grid>

            <Grid item xs={12}>
              <StyledTextField
                fullWidth
                label="What's your story about?"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                multiline
                rows={3}
              />
            </Grid>

            <Grid item xs={12}>
              <StyledButton
                type="submit"
                fullWidth
                disabled={loading || !selectedFiles.length || !prompt}
                startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <StoryIcon />}
              >
                {loading ? 'Generating Story...' : 'Generate Story'}
              </StyledButton>
            </Grid>
          </Grid>
        </form>
      </UploadZone>

      {/* Uploaded Videos Grid */}
      {uploadedVideos.length > 0 && (
        <Fade in={true}>
          <Box>
            <Typography 
              variant="h5" 
              sx={{ 
                color: 'white',
                mb: 3,
                display: 'flex',
                alignItems: 'center',
                gap: 1
              }}
            >
              <MovieIcon /> Uploaded Videos
            </Typography>
            <Grid container spacing={3}>
              {uploadedVideos.map((video, index) => (
                <Grid item xs={12} sm={6} md={4} key={`input-${index}`}>
                  <VideoCard>
                    <Box sx={{ position: 'relative' }}>
                      <Box sx={{ 
                        pt: '56.25%', 
                        position: 'relative',
                        bgcolor: 'black' 
                      }}>
                        {video.thumbnail && (
                          <CardMedia
                            component="img"
                            image={video.thumbnail}
                            alt={video.name}
                            sx={{
                              position: 'absolute',
                              top: 0,
                              left: 0,
                              width: '100%',
                              height: '100%',
                              objectFit: 'contain'
                            }}
                          />
                        )}
                        <IconButton
                          sx={{
                            position: 'absolute',
                            top: '50%',
                            left: '50%',
                            transform: 'translate(-50%, -50%)',
                            bgcolor: 'rgba(0, 0, 0, 0.5)',
                            '&:hover': {
                              bgcolor: 'rgba(0, 0, 0, 0.7)'
                            }
                          }}
                        >
                          <PlayArrowIcon sx={{ color: 'white', fontSize: 40 }} />
                        </IconButton>
                      </Box>
                    </Box>
                    <CardContent>
                      <Typography variant="subtitle1" sx={{ color: 'white' }}>
                        {video.name}
                      </Typography>
                      <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                        {(video.size / (1024 * 1024)).toFixed(2)} MB
                      </Typography>
                    </CardContent>
                  </VideoCard>
                </Grid>
              ))}
            </Grid>
          </Box>
        </Fade>
      )}

      {/* Results Section */}
      {result && (
        <Fade in={true}>
          <Grid container spacing={4}>
            <Grid item xs={12} md={6}>
              <VideoCard>
                <CardContent>
                  <Typography 
                    variant="h6" 
                    sx={{ 
                      color: 'white',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 1,
                      mb: 2
                    }}
                  >
                    <StoryIcon /> Generated Story
                  </Typography>
                  <Typography 
                    variant="body1" 
                    sx={{ 
                      color: 'rgba(255, 255, 255, 0.9)',
                      whiteSpace: 'pre-wrap'
                    }}
                  >
                    {result.story}
                  </Typography>
                </CardContent>
              </VideoCard>
            </Grid>

            <Grid item xs={12} md={6}>
              <VideoCard>
                <CardContent>
                  <Typography 
                    variant="h6" 
                    sx={{ 
                      color: 'white',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 1,
                      mb: 2
                    }}
                  >
                    <MovieIcon /> Generated Video
                  </Typography>
                  {result.videoUrl && (
                    <Box sx={{ 
                      position: 'relative',
                      pt: '56.25%',
                      bgcolor: 'black',
                      borderRadius: 1,
                      overflow: 'hidden'
                    }}>
                      <video
                        controls
                        poster={result.thumbnailUrl}
                        style={{
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          width: '100%',
                          height: '100%',
                          objectFit: 'contain'
                        }}
                        key={result.videoUrl}
                      >
                        <source src={result.videoUrl} type="video/mp4" />
                      </video>
                    </Box>
                  )}
                </CardContent>
              </VideoCard>
            </Grid>
          </Grid>
        </Fade>
      )}
    </StyledContainer>
  );
};

export default VideoUpload; 