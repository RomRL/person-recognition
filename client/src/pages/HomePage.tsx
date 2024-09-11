import React, { useState, useEffect } from 'react';
import {
  Container,
  Box,
  Paper,
  Button,
  CircularProgress,
  Typography,
  IconButton,
  Tooltip,
} from '@mui/material';
import { Search, HelpOutline } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import Header from '../components/Header';
import Footer from '../components/Footer';
import StepperComponent from '../components/StepperComponent';
import AlertDialog from '../components/AlertDialog';
import PersonDetailsForm from '../components/PersonDetailsForm';
import PhotoUploader from '../components/PhotoUploader';
import VideoUploader from '../components/VideoUploader';
import UploadHelp from '../components/UploadHelp';

const HomePage: React.FC = () => {
  // State variables to manage photos, video, and other UI states.
  const [photos, setPhotos] = useState<File[]>([]); // Holds the uploaded photos as an array of File objects
  const [video, setVideo] = useState<File | null>(null); // Holds the uploaded video as a File object or null
  const [isLoading, setIsLoading] = useState<boolean>(false); // Boolean to indicate loading state during processing
  const [personDetails, setPersonDetails] = useState({ id: '', name: '', age: '', height: '' }); // Object to store person details
  const [dialogOpen, setDialogOpen] = useState<boolean>(false); // Boolean to control the visibility of the ID required dialog
  const [alertDialogOpen, setAlertDialogOpen] = useState<boolean>(false); // Boolean to control the visibility of the photos and video required alert dialog
  const [photoInputKey, setPhotoInputKey] = useState<number>(0); // Key to reset photo input field for re-uploading
  const [videoInputKey, setVideoInputKey] = useState<number>(0); // Key to reset video input field for re-uploading
  const [helpDialogOpen, setHelpDialogOpen] = useState<boolean>(false); // Boolean to control the visibility of the help dialog
  const [activeTab, setActiveTab] = useState<number>(0); // (Unused) State variable for managing active tab
  const navigate = useNavigate(); // React Router's useNavigate hook to navigate between pages

  /**
   * useEffect hook to retrieve saved data from localStorage when the component mounts.
   * Fetches photos, video, and person details from localStorage if they exist and sets the state accordingly.
   */
  useEffect(() => {
    const savedPhotos = localStorage.getItem('photos');
    const savedVideo = localStorage.getItem('video');
    const savedPersonDetails = localStorage.getItem('personDetails');

    if (savedPhotos) {
      const photoArray = JSON.parse(savedPhotos);
      const loadedPhotos = photoArray.map(async (photo: any) => {
        const response = await fetch(photo.fileUrl);
        const blob = await response.blob();
        return new File([blob], photo.name, { type: photo.type, lastModified: photo.lastModified });
      });
      Promise.all(loadedPhotos).then(files => setPhotos(files));
    }

    if (savedVideo) {
      const videoData = JSON.parse(savedVideo);
      fetch(videoData.fileUrl)
        .then(res => res.blob())
        .then(blob => setVideo(new File([blob], videoData.name, { type: videoData.type, lastModified: videoData.lastModified })));
    }

    if (savedPersonDetails) {
      setPersonDetails(JSON.parse(savedPersonDetails));
    }
  }, []);

  /**
   * Handles the change event when photos are uploaded.
   * 
   * @param {React.ChangeEvent<HTMLInputElement>} event - The input change event triggered by photo upload.
   */
  const handlePhotoChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setPhotos([...photos, ...Array.from(event.target.files)]); // Append new photos to the existing array
      setPhotoInputKey(photoInputKey + 1);  // Reset input key to allow re-uploading
    }
  };

  /**
   * Handles the change event when a video is uploaded.
   * 
   * @param {React.ChangeEvent<HTMLInputElement>} event - The input change event triggered by video upload.
   */
  const handleVideoChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setVideo(event.target.files[0]); // Set the uploaded video in state
      setVideoInputKey(videoInputKey + 1);  // Reset input key to allow re-uploading
    }
  };

  /**
   * Removes a specific photo from the array based on its index.
   * 
   * @param {number} index - The index of the photo to be removed from the array.
   */
  const handleRemovePhoto = (index: number) => {
    const newPhotos = [...photos];
    newPhotos.splice(index, 1);
    setPhotos(newPhotos);
  };

  /**
   * Removes the uploaded video.
   */
  const handleRemoveVideo = () => {
    setVideo(null);
    setVideoInputKey(videoInputKey + 1);  // Reset input key to allow re-uploading
  };

  /**
   * Handles changes in the person details form fields.
   * 
   * @param {string} field - The field in the personDetails object to be updated (e.g., 'id', 'name').
   * @param {string} value - The new value to be set in the specified field.
   */
  const handlePersonDetailsChange = (field: string, value: string) => {
    setPersonDetails({ ...personDetails, [field]: value }); // Update the specific field in personDetails
  };

  /**
   * Handles the submit event when the "Find Person" button is clicked.
   * Validates the form, saves data to localStorage, and navigates to the loading page.
   */
  const handleSubmit = () => {
    if (!personDetails.id) {
      setDialogOpen(true); // Show dialog if the ID is missing
      return;
    }

    if (photos.length === 0 || !video) {
      setAlertDialogOpen(true); // Show alert if photos or video are missing
      return;
    }

    // Save data to localStorage
    localStorage.setItem('photos', JSON.stringify(photos.map(photo => ({
      name: photo.name,
      lastModified: photo.lastModified,
      size: photo.size,
      type: photo.type,
      fileUrl: URL.createObjectURL(photo),
    }))));

    localStorage.setItem('video', JSON.stringify({
      name: video.name,
      lastModified: video.lastModified,
      size: video.size,
      type: video.type,
      fileUrl: URL.createObjectURL(video),
    }));

    localStorage.setItem('personDetails', JSON.stringify(personDetails));

    // Navigate to the loading page with the uploaded data
    navigate('/loading', { state: { photos, video, personDetails} });
  };

  /**
   * Closes the ID required dialog.
   */
  const handleCloseDialog = () => {
    setDialogOpen(false);
  };

  /**
   * Closes the photos and video required alert dialog.
   */
  const handleCloseAlertDialog = () => {
    setAlertDialogOpen(false);
  };

  /**
   * Opens the help dialog when the help button is clicked.
   */
  const handleHelpButtonClick = () => {
    setHelpDialogOpen(true);
  };

  /**
   * Closes the help dialog.
   */
  const handleCloseHelpDialog = () => {
    setHelpDialogOpen(false);
  };

  return (
    <Box minHeight="100vh">
      <Header />
      <Container sx={{p:3}}>
        <StepperComponent activeStep={0}/>
        <Paper sx={{ p: 3, mt: 2 }}>
          <Typography variant="h5" align="center" sx={{mb: 2}} >Who are we looking for?</Typography>
          <PersonDetailsForm
            personDetails={personDetails}
            handlePersonDetailsChange={handlePersonDetailsChange}
          />
          <PhotoUploader
            photos={photos}
            handlePhotoChange={handlePhotoChange}
            handleRemovePhoto={handleRemovePhoto}
            photoInputKey={photoInputKey}
          />
        </Paper>
        <Paper sx={{ p: 3, mt: 3 }}>
          <Typography variant="h5" align="center" sx={{mb: 2}} >Where should we look?</Typography>
          <VideoUploader
            video={video}
            handleVideoChange={handleVideoChange}
            handleRemoveVideo={handleRemoveVideo}
            videoInputKey={videoInputKey}
          />
        </Paper>
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
          <Button
            variant="contained"
            color="primary"
            startIcon={isLoading ? <CircularProgress size={20} color="inherit" /> : <Search />}
            onClick={handleSubmit}
            disabled={isLoading}
          >
            Find Person
          </Button>
        </Box>
        <Box sx={{ position: 'fixed', top: 70, right: 16, zIndex:1000}}>
          <Tooltip title="Help">
            <IconButton color="primary" onClick={handleHelpButtonClick}>
              <HelpOutline />
            </IconButton>
          </Tooltip>
        </Box>
      </Container>
      <AlertDialog
        open={dialogOpen}
        onClose={handleCloseDialog}
        title="ID Required"
        description="Please fill in the ID field before submitting the form."
      />
      <AlertDialog
        open={alertDialogOpen}
        onClose={handleCloseAlertDialog}
        title="Photos and Video Required"
        description="Please upload at least one photo and one video before submitting the form."
      />
       <UploadHelp open={helpDialogOpen} onClose={handleCloseHelpDialog} />
    </Box>
  );
};

export default HomePage;
