import React, { useEffect, useRef, useState } from 'react';
import { Box, Container, Typography, LinearProgress } from '@mui/material';
import { useNavigate, useLocation } from 'react-router-dom';
import Header from '../components/Header';
import Footer from '../components/Footer';
import StepperComponent from '../components/StepperComponent';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';
import AlertDialog from '../components/AlertDialog'; // Import AlertDialog

// Server address where the backend is hosted
const serverAddress = "http://localhost:8000";

const LoadingPage: React.FC = () => {
  // React Router's useNavigate hook to navigate between pages
  const navigate = useNavigate();
  // useLocation hook to access the state passed through the navigation
  const location = useLocation();
  
  // Destructure the state received from the navigation to extract photos, video, and personDetails
  const { photos, video, personDetails } = location.state as {
    photos: File[], 
    video: File, 
    personDetails: { id: string, name: string, age: string, height: string }
  };

  // State to control the visibility of the error dialog
  const [errorDialogOpen, setErrorDialogOpen] = useState(false);
  // State to hold the error message to be displayed in the dialog
  const [errorMessage, setErrorMessage] = useState('');

  // Ref to track if an error has already been displayed to avoid multiple error dialogs
  const errorDisplayed = useRef(false);
  // Ref to ensure the useEffect only runs once
  const effectRan = useRef(false);

  /**
   * Closes the error dialog and navigates back to the homepage.
   */
  const handleErrorClose = () => {
    setErrorDialogOpen(false);
    navigate('/'); // Navigate back to the homepage
  };

  /**
   * Clears blob URLs for photos and video from localStorage to free up memory.
   */
  const clearBlobUrls = () => {
    const savedPhotos = localStorage.getItem('photos');
    const savedVideo = localStorage.getItem('video');

    if (savedPhotos) {
      const photoArray = JSON.parse(savedPhotos);
      photoArray.forEach((photo: any) => URL.revokeObjectURL(photo.fileUrl));
    }

    if (savedVideo) {
      const videoData = JSON.parse(savedVideo);
      URL.revokeObjectURL(videoData.fileUrl);
    }
  };

  useEffect(() => {
    if (effectRan.current || errorDisplayed.current) return;
    effectRan.current = true;

    // Generate a unique ID for the current processing session
    const runningId = uuidv4();

    /**
     * Asynchronous function to handle the video processing.
     * It sends the photos and video to the backend server for processing.
     * In case of an error, it calls the handleError function to display an error dialog.
     */
    const processVideo = async () => {
      try {
        const photoFormData = new FormData();
        photos.forEach(photo => {
          photoFormData.append('files', photo);
        });
        photoFormData.append('user_json', JSON.stringify(personDetails));

        const videoFormData = new FormData();
        videoFormData.append('file', video);

        // Set reference image(s) by sending the photos to the server
        await axios.post(`${serverAddress}/set_reference_image/?uuid=${personDetails.id}`, photoFormData, {
          headers: {
            'Content-Type': 'multipart/form-data',
            'accept': 'application/json',
          },
        }).catch((error) => {
          handleError("Failed to set reference image.", error);
          throw error;
        });

        // Send the video to the server for processing and receive the annotated video
        const response = await axios.post(`${serverAddress}/detect_and_annotate/?uuid=${personDetails.id}&running_id=${runningId}&similarity_threshold=60`, videoFormData, {
          headers: {
            'Content-Type': 'multipart/form-data',
            'accept': 'application/json',
          },
          responseType: 'blob', // Ensure the response is of type 'blob'
        }).catch((error) => {
          handleError("Video processing failed.", error);
          throw error;
        });

        if (response?.status === 200) {
          const videoBlob = response.data;
          const videoUrl = URL.createObjectURL(videoBlob);

          // Fetch the detected frames from the server
          const framesResponse = await axios.get(`${serverAddress}/get_detected_frames/?uuid=${personDetails.id}&running_id=${runningId}`, {
            headers: {
              'accept': 'application/json',
            },
          }).catch((error) => {
            handleError("Failed to fetch detected frames.", error);
            throw error;
          });

          const framesData = framesResponse?.data;

          if (framesData?.status === "success") {
            const detectedFrames = framesData.detected_frames;

            // Clear local storage on success
            clearBlobUrls();
            localStorage.removeItem('photos');
            localStorage.removeItem('video');
            localStorage.removeItem('personDetails');

            // Navigate to the result page with video and detected frames
            navigate('/result', { state: { detectedFrames, videoUrl } });
          } else {
            throw new Error('Failed to fetch detected frames');
          }
        } else {
          throw new Error("Video processing failed with an unexpected status.");
        }
      } catch (error) {
        if (!errorDisplayed.current) {
          console.error('Error processing video:', error);
          handleError("An error occurred during video processing.", error);
        }
      }
    };

    /**
     * Handles errors during the video processing.
     * Displays a detailed error message to the user in a dialog.
     * 
     * @param {string} userMessage - A user-friendly message describing the error context.
     * @param {any} error - The error object caught during the process.
     */
    const handleError = (userMessage: string, error: any) => {
      if (errorDisplayed.current) return;
      errorDisplayed.current = true;

      let fullErrorMessage = '';

      if (error.message === "Network Error") {
        // Specific message for network errors
        fullErrorMessage = `
          ${userMessage}

          Error: Network Error
          It seems there is a problem with the network connection or the server may be down.
        `;
      } else {
        // Handle other types of errors
        const statusCode = error.response?.status;
        const statusText = error.response?.statusText;
        const errorMessage = error.message;

        fullErrorMessage = `
          ${userMessage}

          Status: ${statusCode} ${statusText}

          Error: ${errorMessage}
        `;
      }

      setErrorMessage(fullErrorMessage);
      setErrorDialogOpen(true);
    };

    // Initiate video processing
    processVideo();
  }, [navigate, photos, video, personDetails.id]);

  return (
    <Box display="flex" flexDirection="column" minHeight="100vh">
      <Header />
      <Container maxWidth="md" style={{ marginTop: '2rem', flexGrow: 1, textAlign: 'center' }}>
        <StepperComponent activeStep={1} />
        <Typography component="h1" variant="h5" gutterBottom>
          Processing
        </Typography>
        <Box
          component="img"
          src={"assets/finding.gif"}
          alt="Loading"
          sx={{
            maxWidth: '100%',
            height: 'auto',
            margin: '20px auto',
          }}
        />
        <LinearProgress />
        <Typography variant="body1" color="textSecondary" style={{ marginTop: '1rem' }}>
          We are looking for the person in the video. This might take a few moments.
        </Typography>
      </Container>
      <Footer />

      <AlertDialog
        open={errorDialogOpen}
        onClose={handleErrorClose}
        title="Error"
        description={errorMessage}
      />
    </Box>
  );
};

export default LoadingPage;
