import React from 'react';
import { Grid, TextField } from '@mui/material';

interface PersonDetailsFormProps {
  personDetails: { id: string; name: string; age: string; height: string };
  handlePersonDetailsChange: (field: string, value: string) => void;
}

/**
 * PersonDetailsForm component that renders a form for entering personal details.
 * The form includes fields for ID, Name, Age, and Height, with ID being required.
 * 
 * @component
 * @param {Object} personDetails - An object containing the details of the person (ID, Name, Age, Height).
 * @param {(field: string, value: string) => void} handlePersonDetailsChange - Function to handle the change in form fields. 
 * @returns {JSX.Element} The rendered PersonDetailsForm component.
 */
const PersonDetailsForm: React.FC<PersonDetailsFormProps> = ({ personDetails, handlePersonDetailsChange }) => {
  return (
    <Grid container spacing={2} alignItems="center">
      <Grid item xs={12} sm={3}>
        <TextField
          label="ID"
          value={personDetails.id}
          onChange={(e) => handlePersonDetailsChange('id', e.target.value)}
          fullWidth
          size="small"
          required
        />
      </Grid>
      <Grid item xs={12} sm={3}>
        <TextField
          label="Name (Optional)"
          value={personDetails.name}
          onChange={(e) => handlePersonDetailsChange('name', e.target.value)}
          fullWidth
          size="small"
        />
      </Grid>
      <Grid item xs={12} sm={3}>
        <TextField
          label="Age (Optional)"
          value={personDetails.age}
          onChange={(e) => handlePersonDetailsChange('age', e.target.value)}
          fullWidth
          size="small"
        />
      </Grid>
      <Grid item xs={12} sm={3}>
        <TextField
          label="Height (Optional)"
          value={personDetails.height}
          onChange={(e) => handlePersonDetailsChange('height', e.target.value)}
          fullWidth
          size="small"
        />
      </Grid>
    </Grid>
  );
};

export default PersonDetailsForm;
