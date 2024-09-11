import React from 'react';
import { Stepper, Step, StepLabel } from '@mui/material';

interface StepperComponentProps {
  activeStep: number;
}

/**
 * StepperComponent displays a three-step progress indicator.
 * The steps are: "Upload Photos and Video", "Processing", and "Result".
 * 
 * @component
 * @param {number} activeStep - The index of the currently active step (0-based index).
 * @returns {JSX.Element} The rendered StepperComponent.
 */
const StepperComponent: React.FC<StepperComponentProps> = ({ activeStep }) => {
  return (
    <Stepper activeStep={activeStep} alternativeLabel style={{ marginBottom: '2rem' }}>
      <Step>
        <StepLabel>Upload Photos and Video</StepLabel>
      </Step>
      <Step>
        <StepLabel>Processing</StepLabel>
      </Step>
      <Step>
        <StepLabel>Result</StepLabel>
      </Step>
    </Stepper>
  );
};

export default StepperComponent;
