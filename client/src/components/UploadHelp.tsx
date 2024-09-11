import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab,
  Box,
  Typography,
  Button,
  Skeleton
} from '@mui/material';

interface UploadHelpProps {
  open: boolean;
  onClose: () => void;
}

/**
 * UploadHelp component provides a help dialog with instructions on how to use the upload functionality.
 * The dialog includes tabs for different steps, each containing relevant instructions and an example video.
 * 
 * @component
 * @param {boolean} open - Controls whether the dialog is open or closed.
 * @param {() => void} onClose - Function to close the dialog.
 * @returns {JSX.Element} The rendered UploadHelp component.
 */
const UploadHelp: React.FC<UploadHelpProps> = ({ open, onClose }) => {
  // State to track the active tab in the dialog
  const [activeTab, setActiveTab] = useState<number>(0);
  // State to track whether the video has loaded
  const [videoLoaded, setVideoLoaded] = useState<boolean>(false);

  /**
   * Handles the event when a tab is selected, resetting the videoLoaded state.
   * 
   * @param {React.SyntheticEvent} event - The event triggered by selecting a tab.
   * @param {number} newValue - The index of the selected tab.
   */
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
    setVideoLoaded(false); // Reset video loaded state when switching tabs
  };

  /**
   * Handles the event when the video has loaded, setting the videoLoaded state to true.
   */
  const handleVideoLoad = () => {
    setVideoLoaded(true);
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle align="center">
        User Help 
      </DialogTitle>
      <DialogContent>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={activeTab} onChange={handleTabChange} variant="fullWidth">
            <Tab label="Step 1" />
            <Tab label="Step 2" />
            <Tab label="Step 3" />
          </Tabs>
        </Box>
        <Box p={3}>
          {activeTab === 0 && (
            <div>
              <Typography variant="h6">Fill User Details</Typography>
              <Typography variant="body1">The ID is the only required field to fill, but feel free to fill all the information you have.</Typography>
              {!videoLoaded && <Skeleton variant="rectangular" width="100%" height={200} />}
              <video
                src="assets/User_Details.mp4"
                width={"100%"}
                controls
                onLoadedData={handleVideoLoad}
                style={{ display: videoLoaded ? 'block' : 'none' }}
              />
            </div>
          )}
          {activeTab === 1 && (
            <div>
              <Typography variant="h6">Upload Photos</Typography>
              <Typography variant="body1">Upload the photos of the person you want us to look for in the video.<br /> Make sure that the photos you upload show only the person you want us to look for without any other person beside them.</Typography>
              {!videoLoaded && <Skeleton variant="rectangular" width="100%" height={200} />}
              <video
                src="assets/Images_Upload.mp4"
                width={"100%"}
                controls
                onLoadedData={handleVideoLoad}
                style={{ display: videoLoaded ? 'block' : 'none' }}
              />
            </div>
          )}
          {activeTab === 2 && (
            <div>
              <Typography variant="h6">Upload Video</Typography>
              <Typography variant="body1">Upload the video you want us to find the person in, and we will return to you the same video with boxes surrounding the person of interest.</Typography>
              {!videoLoaded && <Skeleton variant="rectangular" width="100%" height={200} />}
              <video
                src="assets/Video_Upload.mp4"
                width={"100%"}
                controls
                onLoadedData={handleVideoLoad}
                style={{ display: videoLoaded ? 'block' : 'none' }}
              />
            </div>
          )}
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} color="primary">
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default UploadHelp;
