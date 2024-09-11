import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography
} from '@mui/material';

interface AlertDialogProps {
  open: boolean;
  onClose: () => void;
  title: string;
  description: string;
}

/**
 * AlertDialog component that displays a modal dialog with a title, description, and an OK button.
 * The description text can be split into multiple lines, each rendered separately.
 * 
 * @component
 * @param {boolean} open - Controls whether the dialog is open or closed.
 * @param {() => void} onClose - Function to be called when the dialog is closed.
 * @param {string} title - The title of the dialog.
 * @param {string} description - The description text displayed in the dialog. Supports multiple lines by splitting on '\n'.
 * @returns {JSX.Element} The rendered AlertDialog component.
 */
const AlertDialog: React.FC<AlertDialogProps> = ({ open, onClose, title, description }) => {
  // Format the description to support multiple lines by splitting on '\n'
  const formattedDescription = description.split('\\n').map((line, index) => (
    <React.Fragment key={index}>
      {line}
      <br />
    </React.Fragment>
  ));

  return (
    <Dialog
      open={open}
      onClose={onClose}
      aria-labelledby="alert-dialog-title"
      aria-describedby="alert-dialog-description"
    >
      <DialogTitle id="alert-dialog-title">{title}</DialogTitle>
      <DialogContent>
        <Typography variant="body1">{formattedDescription}</Typography>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} color="primary" autoFocus>
          OK
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default AlertDialog;
