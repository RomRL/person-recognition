import React from 'react';
import { Box, Grid, IconButton, Button } from '@mui/material';
import { PhotoCamera, Close } from '@mui/icons-material';

interface PhotoUploaderProps {
  photos: File[];
  handlePhotoChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  handleRemovePhoto: (index: number) => void;
  photoInputKey: number;
}

/**
 * PhotoUploader component that allows users to upload and preview photos.
 * It also provides functionality to remove uploaded photos.
 * 
 * @component
 * @param {File[]} photos - An array of photo files that have been uploaded.
 * @param {(event: React.ChangeEvent<HTMLInputElement>) => void} handlePhotoChange - Function to handle photo upload events.
 * @param {(index: number) => void} handleRemovePhoto - Function to handle the removal of a specific photo based on its index.
 * @param {number} photoInputKey - A key to reset the file input field, allowing the same file to be re-uploaded.
 * @returns {JSX.Element} The rendered PhotoUploader component.
 */
const PhotoUploader: React.FC<PhotoUploaderProps> = ({ photos, handlePhotoChange, handleRemovePhoto, photoInputKey }) => {
  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
        <label htmlFor="upload-photo">
          <input
            accept="image/*"
            style={{ display: 'none' }}
            id="upload-photo"
            type="file"
            multiple
            onChange={handlePhotoChange}
            key={photoInputKey}
          />
          <Button
            component="span"
            variant="outlined"
            startIcon={<PhotoCamera />}
            sx={{ whiteSpace: 'nowrap' }}
          >
            Upload Photo
          </Button>
        </label>
      </Box>
      <Grid container spacing={2} sx={{ mt: 2 }}>
        {photos.map((photo, index) => (
          <Grid item xs={3} key={index} position="relative">
            <img
              src={URL.createObjectURL(photo)}
              alt={`photo-${index}`}
              style={{
                width: '100%',
                height: 'auto',
                objectFit: 'cover',
                borderRadius: '4px',
              }}
            />
            <IconButton
              onClick={() => handleRemovePhoto(index)}
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
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default PhotoUploader;
