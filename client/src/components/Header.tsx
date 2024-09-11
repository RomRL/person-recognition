import React from 'react';
import { AppBar, Toolbar, Typography, Button, IconButton } from '@mui/material';
import { PhotoCamera } from '@mui/icons-material';

/**
 * Header component that displays a navigation bar at the top of the page.
 * The header includes a logo, the title of the application, and navigation buttons.
 * 
 * @component
 * @returns {JSX.Element} The rendered Header component.
 */
const Header: React.FC = () => (
  <AppBar
    position="static"
    sx={{
      background: 'linear-gradient(45deg, blue, purple)',
      color: 'white',
    }}
  >
    <Toolbar>
      <IconButton edge="start" color="inherit" aria-label="menu" href='/'>
        <PhotoCamera />
      </IconButton>
      <Typography variant="h6" style={{ flexGrow: 1 }}>
        FindPerson
      </Typography>
      <Button color="inherit" href="/">Home</Button>
      <Button color="inherit" href='/about'>About</Button>
      <Button color="inherit" href='/contact'>Contact</Button>
    </Toolbar>
  </AppBar>
);

export default Header;
