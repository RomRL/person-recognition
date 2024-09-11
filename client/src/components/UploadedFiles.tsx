import React from 'react';
import { Grid, Box, Card, CardMedia, IconButton, Typography, Paper } from '@mui/material';
import { Close, Panorama } from '@mui/icons-material';

interface UploadedFilesProps {
  photos: File[];
  video: File | null;
  onRemovePhoto: (index: number) => void;
  onRemoveVideo: () => void;
}

/**
 * UploadedFiles component displays a list of uploaded photos and video, 
 * allowing the user to remove any of the uploaded files.
 * 
 * @component
 * @param {File[]} photos - An array of uploaded photo files.
 * @param {File | null} video - The uploaded video file, or null if no video has been uploaded.
 * @param {(index: number) => void} onRemovePhoto - Function to remove a specific photo based on its index.
 * @param {() => void} onRemoveVideo - Function to remove the uploaded video.
 * @returns {JSX.Element} The rendered UploadedFiles component.
 */
const UploadedFiles: React.FC<UploadedFilesProps> = ({ photos, video, onRemovePhoto, onRemoveVideo }) => {
  // Generate placeholder cards if less than 3 photos are uploaded
  const placeholders = Array.from({ length: 3 - photos.length }).map((_, index) => (
    <Grid item xs={12} sm={6} md={4} key={`placeholder-${index}`}>
      <Paper
        sx={{
          width: '100%',
          height: 140,
          backgroundColor: 'rgba(0,0,0,0.1)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Panorama sx={{ color: 'rgba(0,0,0,0.3)', fontSize: 40 }} />
      </Paper>
    </Grid>
  ));

  return (
    <div>
      <Typography variant="h6" gutterBottom>
        Uploaded Files
      </Typography>
      <Grid container spacing={2}>
        {photos.map((photo, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
            <Box position="relative">
              <Card sx={{ maxWidth: 345 }}>
                <CardMedia
                  component="img"
                  alt={`Photo ${index + 1}`}
                  height="140"
                  image={URL.createObjectURL(photo)}
                />
              </Card>
              <IconButton
                onClick={() => onRemovePhoto(index)}
                size="small"
                sx={{
                  position: 'absolute',
                  top: 8,
                  left: 8,
                  bgcolor: 'rgba(255,255,255,0.8)',
                }}
              >
                <Close />
              </IconButton>
            </Box>
          </Grid>
        ))}
        {photos.length < 3 && placeholders}
        {video && (
          <Grid item xs={12}>
            <Box position="relative">
              <Card sx={{ maxWidth: 345 }}>
                <CardMedia
                  component="video"
                  controls
                  src={URL.createObjectURL(video)}
                  title="Uploaded Video"
                />
              </Card>
              <IconButton
                onClick={onRemoveVideo}
                size="small"
                sx={{
                  position: 'absolute',
                  top: 8,
                  left: 8,
                  bgcolor: 'rgba(255,255,255,0.8)',
                }}
              >
                <Close />
              </IconButton>
            </Box>
          </Grid>
        )}
      </Grid>
    </div>
  );
};

export default UploadedFiles;
