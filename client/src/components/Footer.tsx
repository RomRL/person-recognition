import React from 'react';
import { Box, Container, Typography } from '@mui/material';

/**
 * Footer component that displays a footer section at the bottom of the page.
 * The footer includes links to the Privacy Policy and Terms of Service.
 * 
 * @component
 * @returns {JSX.Element} The rendered Footer component.
 */
const Footer: React.FC = () => (
  <Box
    component="footer"
    py={2}
    bgcolor="grey.200"
    sx={{ 
      width: '100%',
      textAlign: 'center',
      marginBottom: 0,
    }}
  >
    <Container maxWidth="md">
      <Typography variant="body2" color="textSecondary">
        Privacy Policy | Terms of Service
      </Typography>
    </Container>
  </Box>
);

export default Footer;
