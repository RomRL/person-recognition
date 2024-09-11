import React from 'react';
import { Box, Button, IconButton } from '@mui/material';
import { VideoCameraBack, Close } from '@mui/icons-material';

interface VideoUploaderProps {
  video: File | null;
  handleVideoChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  handleRemoveVideo: () => void;
  videoInputKey: number;
}

/**
 * VideoUploader component allows users to upload and preview a video file.
 * It also provides functionality to remove the uploaded video.
 * 
 * @component
 * @param {File | null} video - The uploaded video file, or null if no video has been uploaded.
 * @param {(event: React.ChangeEvent<HTMLInputElement>) => void} handleVideoChange - Function to handle the video upload event.
 * @param {() => void} handleRemoveVideo - Function to handle the removal of the uploaded video.
 * @param {number} videoInputKey - A key to reset the file input field, allowing the same file to be re-uploaded.
 * @returns {JSX.Element} The rendered VideoUploader component.
 */
const VideoUploader: React.FC<VideoUploaderProps> = ({ video, handleVideoChange, handleRemoveVideo, videoInputKey }) => {
  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
        <label htmlFor="upload-video">
          <input
            accept="video/*"
            style={{ display: 'none' }}
            id="upload-video"
            type="file"
            onChange={handleVideoChange}
            key={videoInputKey}
          />
          <Button
            component="span"
            variant="outlined"
            startIcon={<VideoCameraBack />}
            sx={{ whiteSpace: 'nowrap' }}
          >
            Upload Video
          </Button>
        </label>
      </Box>
      {video && (
        <Box mt={2} display="flex" justifyContent="center" alignItems="center" position="relative">
          <video width="100%" height="auto" controls>
            <source src={URL.createObjectURL(video)} type="video/mp4" />
          </video>
          <IconButton
            onClick={handleRemoveVideo}
            style={{
              position: 'absolute',
              top: '0.5rem',
              left: '0.5rem',
              background: 'rgba(255, 255, 255, 0.8)',
              transform: 'scale(0.8)',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.opacity = '0.7')}
            onMouseLeave={(e) => (e.currentTarget.style.opacity = '1')}
          >
            <Close />
          </IconButton>
        </Box>
      )}
    </Box>
  );
};

export default VideoUploader;
