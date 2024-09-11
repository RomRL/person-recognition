import React, { useState } from 'react';
import { Box, Slider } from '@mui/material';
import { styled } from '@mui/system';
import { useSwipeable } from 'react-swipeable';

interface Detection {
  cropped_image: string;
  similarity: number;
}

interface CustomCarouselProps {
  frameKeys: string[];
  detectedFrames: { [key: string]: Detection };
  onImageClick: (frameIndex: number) => void;
}

// Styled component for the main carousel container
const CarouselContainer = styled(Box)(({ theme }) => ({
  position: 'relative',
  width: '100%',
  height: '500px',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  overflow: 'hidden',
}));

// Styled component for individual images in the carousel
const CarouselImage = styled(Box)<{ active: boolean }>(({ active }) => ({
  position: 'absolute',
  transition: 'all 0.3s ease',
  opacity: active ? 1 : 0.5,
  transform: active ? 'scale(1)' : 'scale(0.8)',
  zIndex: active ? 2 : 1,
  cursor: 'pointer',
}));

// Styled component for the navigation images
const NavImage = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: '50%',
  transform: 'translateY(-50%)',
  width: '80px',
  height: '80px',
  borderRadius: '10px',
  overflow: 'hidden',
  cursor: 'pointer',
  zIndex: 3,
  opacity: 0.7,
  transition: 'opacity 0.3s ease',
  '&:hover': {
    opacity: 1,
  },
  [theme.breakpoints.down('sm')]: {
    width: '50px',
    height: '50px',
    top: '80%', // Move it down so it doesn't overlap the main image
    transform: 'translateY(0)', // Reset the transform
  },
}));

// Styled component for the slider container
const SliderContainer = styled(Box)(({ theme }) => ({
  position: 'absolute',
  bottom: 10, // Move it down slightly by reducing the bottom position
  left: '50%',
  transform: 'translateX(-50%)',
  width: '80%',
  zIndex: 3,
  marginTop: '30px', // Add margin to move it further down
}));

/**
 * CustomCarousel component that displays a carousel of images, each representing a detected frame.
 * The carousel supports navigation through arrow clicks or a slider.
 * 
 * @component
 * @param {string[]} frameKeys - An array of keys representing the detected frames.
 * @param {{ [key: string]: Detection }} detectedFrames - An object containing detection data for each frame.
 * @param {(frameIndex: number) => void} onImageClick - Callback function that is triggered when an image is clicked. The index of the frame is passed as a parameter.
 * @returns {JSX.Element} The rendered CustomCarousel component.
 */
const CustomCarousel: React.FC<CustomCarouselProps> = ({ frameKeys, detectedFrames, onImageClick }) => {
  // State to track the current index of the active image in the carousel
  const [currentIndex, setCurrentIndex] = useState(0);

  /**
   * Handles the event when the previous navigation arrow is clicked.
   * Updates the current index to the previous frame or wraps around to the last frame.
   */
  const handlePrev = () => {
    setCurrentIndex((prevIndex) => (prevIndex > 0 ? prevIndex - 1 : frameKeys.length - 1));
  };

  /**
   * Handles the event when the next navigation arrow is clicked.
   * Updates the current index to the next frame or wraps around to the first frame.
   */
  const handleNext = () => {
    setCurrentIndex((prevIndex) => (prevIndex < frameKeys.length - 1 ? prevIndex + 1 : 0));
  };

  /**
   * Handles the change event when the slider is used to navigate the carousel.
   * 
   * @param {Event} event - The slider change event.
   * @param {number | number[]} newValue - The new value from the slider, representing the index of the selected frame.
   */
  const handleSliderChange = (event: Event, newValue: number | number[]) => {
    setCurrentIndex(newValue as number);
  };

  /**
   * Utility function to get the index of the previous frame in the carousel.
   * 
   * @returns {number} The index of the previous frame.
   */
  const getPrevIndex = () => (currentIndex > 0 ? currentIndex - 1 : frameKeys.length - 1);

  /**
   * Utility function to get the index of the next frame in the carousel.
   * 
   * @returns {number} The index of the next frame.
   */
  const getNextIndex = () => (currentIndex < frameKeys.length - 1 ? currentIndex + 1 : 0);

  // Integrate swipeable hooks
  const handlers = useSwipeable({
    onSwipedLeft: () => handleNext(),
    onSwipedRight: () => handlePrev(),
    trackMouse: true, // Enable mouse swipe events as well
  });

  return (
    <CarouselContainer {...handlers}>
      {frameKeys.map((key, index) => (
        <CarouselImage
          key={index}
          active={index === currentIndex}
          style={{
            display: Math.abs(index - currentIndex) <= 1 ? 'block' : 'none',
          }}
          onClick={() => onImageClick(parseInt(key.replace('frame_', ''), 10))}
        >
          <img
            src={`data:image/jpeg;base64,${(detectedFrames[key] as Detection).cropped_image}`}
            alt={`Detection ${index + 1}`}
            style={{ maxWidth: '100%', maxHeight: '300px', borderRadius: '10px', boxShadow: '0px 4px 15px rgba(0, 0, 0, 0.2)' }}
          />
        </CarouselImage>
      ))}
      <NavImage onClick={handlePrev} style={{ left: 10 }}>
        <img
          src={`data:image/jpeg;base64,${(detectedFrames[frameKeys[getPrevIndex()]] as Detection).cropped_image}`}
          alt="Previous"
          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
        />
      </NavImage>
      <NavImage onClick={handleNext} style={{ right: 10 }}>
        <img
          src={`data:image/jpeg;base64,${(detectedFrames[frameKeys[getNextIndex()]] as Detection).cropped_image}`}
          alt="Next"
          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
        />
      </NavImage>
      <SliderContainer>
        <Slider
          value={currentIndex}
          onChange={handleSliderChange}
          min={0}
          max={frameKeys.length - 1}
          step={1}
          marks
          sx={{
            '& .MuiSlider-thumb': {
              width: 16,
              height: 16,
              backgroundColor: '#00ace6',
            },
            '& .MuiSlider-track': {
              backgroundColor: '#primary',
              height: 8,
            },
            '& .MuiSlider-rail': {
              backgroundColor: 'rgba(255, 255, 255, 0.5)',
              height: 8,
            },
            '& .MuiSlider-mark': {
              backgroundColor: '#fff',
              width: 4,
              height: 4,
              borderRadius: '50%',
            },
          }}
        />
      </SliderContainer>
    </CarouselContainer>
  );
};

export default CustomCarousel;
