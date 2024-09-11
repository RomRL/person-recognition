import React, { useRef, useEffect, useState } from 'react';
import { Box, Container, Typography, Card, CardMedia, CardContent, Grid } from '@mui/material';
import { useLocation } from 'react-router-dom';
import Header from '../components/Header';
import Footer from '../components/Footer';
import StepperComponent from '../components/StepperComponent';
import CustomCarousel from '../components/CustomCarousel';
import AlertDialog from '../components/AlertDialog';


interface Detection {
  cropped_image: string;
  similarity: number;
}

interface UserDetails {
  id: string;
  name: string;
  age: string;
  height: string;
}

interface DetectedFrames {
  frame_per_second: number;
  [key: string]: Detection | UserDetails | number;
}

const ResultPage: React.FC = () => {

  const [dialogOpen, setDialogOpen] = useState<boolean>(false); // Boolean to control the visibility of the no detection found dialog

  // useLocation hook to access the state passed through navigation
  const location = useLocation();

  // Destructure the state to extract detectedFrames and videoUrl
  const { detectedFrames, videoUrl } = location.state as {
    detectedFrames: { [key: string]: Detection | UserDetails }, 
    videoUrl: string 
  };

  // The frames per second value is extracted and incremented by one
  const framePerSecond = (detectedFrames.frame_per_second as unknown) as number + 1;

  // useRef hook to create a reference to the video element
  const videoRef = useRef<HTMLVideoElement>(null);

  /**
   * useEffect hook to reload the video element whenever the video URL changes.
   */
  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.load(); // Ensure the video is reloaded when the component mounts or when the videoUrl changes
    }
  }, [videoUrl]);

  /**
   * Handles the click event on an image in the carousel.
   * Seeks to the corresponding time in the video based on the frame index and pauses the video.
   * 
   * @param {number} frameIndex - The index of the frame that was clicked.
   */
  const handleImageClick = (frameIndex: number) => {
    if (videoRef.current && !isNaN(frameIndex)) {
      const currentTime = frameIndex / framePerSecond; // Calculate the current time in the video
      if (!isNaN(currentTime) && isFinite(currentTime)) {
        videoRef.current.currentTime = currentTime; // Seek to the calculated time
        videoRef.current.pause(); // Pause the video
      } else {
        console.error('Invalid frame index or calculated time:', frameIndex, currentTime);
      }
    } else {
      console.error('Video reference is not valid or frame index is NaN:', videoRef.current, frameIndex);
    }
  };


  /**
   * Closes the no detection found dialog.
   */
  const handleCloseDialog = () => {
    console.log('closing');
    setDialogOpen(false);
    console.log('Dialog open state after closing:', dialogOpen);
  };

  // Extract user details from the detectedFrames object
  const userDetails = detectedFrames.user_details as UserDetails;

  // Filter out the keys that are not related to detected frames
  const frameKeys = Object.keys(detectedFrames).filter(key => key !== 'user_details' && key !== 'frame_per_second');

  useEffect(() => {
    if (frameKeys.length === 0) {
      setDialogOpen(true); // Open the dialog if no detections are found
    }
  }, []); // Only run when frameKeys changes
  

  // Create a filtered version of detectedFrames that only includes Detection types
  const filteredDetectedFrames = Object.keys(detectedFrames).reduce<{ [key: string]: Detection }>((result, key) => {
    if (key !== 'user_details' && key !== 'frame_per_second') {
      result[key] = detectedFrames[key] as Detection;
    }
    return result;
  }, {});

  return (
    <Box display="flex" flexDirection="column" minHeight="100vh">
      <Header />
      <Container maxWidth="md" style={{ marginTop: '2rem', flexGrow: 1 }}>
        <StepperComponent activeStep={2} />
        <Typography component="h1" variant="h5" align="center" gutterBottom>
          Processed Video
        </Typography>
        <Card>
          <CardMedia
            component="video"
            controls
            src={videoUrl}
            ref={videoRef}
            sx={{ borderRadius: 1, boxShadow: 3 }}
          />
        </Card>
        {userDetails && (
          <Card style={{ marginTop: '1rem' }}>
            <CardContent>
              <Typography component="h2" variant="h6" align="center" gutterBottom>
                Person Details
              </Typography>
              <Grid container spacing={2} justifyContent="center">
                {userDetails.id && (
                  <Grid item>
                    <Typography variant="body1">ID: {userDetails.id}</Typography>
                  </Grid>
                )}
                {userDetails.name && (
                  <Grid item>
                    <Typography variant="body1">Name: {userDetails.name}</Typography>
                  </Grid>
                )}
                {userDetails.age && (
                  <Grid item>
                    <Typography variant="body1">Age: {userDetails.age}</Typography>
                  </Grid>
                )}
                {userDetails.height && (
                  <Grid item>
                    <Typography variant="body1">Height: {userDetails.height}</Typography>
                  </Grid>
                )}
              </Grid>
            </CardContent>
          </Card>
        )}
        <Card style={{ marginTop: '1rem' }}>
          <Typography component="h2" variant="h6" align="center" gutterBottom>
            Detections
          </Typography>
          <CardContent>
            {frameKeys.length == 0 ? 
              <Typography variant="body1" align="center"> No detections found in the video. </Typography> :
            <CustomCarousel
              frameKeys={frameKeys}
              detectedFrames={filteredDetectedFrames} // Pass filtered detectedFrames
              onImageClick={handleImageClick}
            />
            } 
          </CardContent>
        </Card>
      </Container>
      <Footer />
      <AlertDialog
        open={dialogOpen}
        onClose={handleCloseDialog}
        title="No Detections Found"
        description="It seems that the person you are looking for is not in the video. \nPlease try again with a different video or add new reference photos."
      />
    </Box>
  );
};

export default ResultPage;
