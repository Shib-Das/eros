//! # Terminal User Interface (TUI) Module
//!
//! This module provides functions for setting up and restoring the terminal
//! for the TUI application. It uses the `crossterm` and `ratatui` crates
//! to create a professional and responsive terminal interface.

use anyhow::Result;
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io;

/// Sets up the terminal for the TUI application.
///
/// This function enables raw mode, enters the alternate screen, and enables mouse capture.
/// It returns a `Terminal` instance that can be used to draw the TUI.
///
/// # Returns
///
/// A `Result` containing the `Terminal` instance, or an error if the setup fails.
pub fn setup_terminal() -> Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    Terminal::new(backend).map_err(Into::into)
}

/// Restores the terminal to its original state.
///
/// This function disables raw mode, leaves the alternate screen, and disables mouse capture.
/// It should be called when the application exits to ensure the terminal is left in a clean state.
///
/// # Arguments
///
/// * `terminal` - A mutable reference to the `Terminal` instance.
///
/// # Returns
///
/// A `Result` indicating whether the restoration was successful.
pub fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    Ok(())
}