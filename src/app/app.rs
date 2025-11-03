//! This module defines the core application logic for the TUI.
//!
//! It manages the application's state, handles user input, and orchestrates the
//! media processing pipeline. The `App` struct is the central component,
//! controlling the UI flow and managing the background processing tasks.

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use image::DynamicImage;
use ratatui::{backend::Backend, Terminal};
use std::{
    io,
    path::{Path, PathBuf},
};
use tokio::sync::mpsc;

use crate::args::V3Model;
use eros::prelude::suggest_media_directories;

use super::ui;
use crate::core::{run_full_process, AppConfig};

/// Represents updates sent from the processing thread to the UI thread.
#[derive(Debug)]
pub enum ProgressUpdate {
    Message(String),
    Progress(f64),
    Error(String),
    ImageProcessed(PathBuf),
    Complete,
}

/// Represents the different screens in the TUI.
#[derive(Debug, PartialEq, Eq)]
pub enum CurrentScreen {
    SuggestingDirs,
    Main,
    Editing,
    Processing,
    Finished,
    Exiting,
}

/// Represents the items in the main menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MenuItem {
    Model,
    InputPath,
    Threshold,
    BatchSize,
    ShowAsciiArt,
    Start,
}

/// The main application struct, holding the state of the TUI.
pub struct App {
    config: AppConfig,
    current_screen: CurrentScreen,
    currently_editing: Option<MenuItem>,
    menu_items: Vec<MenuItem>,
    menu_index: usize,
    input_text: String,
    pub progress: f64,
    pub status_message: String,
    rx: Option<mpsc::Receiver<ProgressUpdate>>,
    pub is_error: bool,
    pub suggested_dirs: Vec<PathBuf>,
    pub selected_dirs: Vec<PathBuf>,
    pub suggestion_index: usize,
    pub show_ascii_art: bool,
    pub current_frame: Option<DynamicImage>,
    pub logs: Vec<String>,
    pub processed_image_paths: Vec<PathBuf>,
    pub current_image_index: usize,
}

impl Default for App {
    fn default() -> Self {
        let suggested_dirs = suggest_media_directories(Path::new(".")).unwrap_or_default();
        Self {
            config: AppConfig {
                model: V3Model::SwinV2,
                input_path: "./images".to_string(),
                threshold: 0.5,
                batch_size: 1,
                show_ascii_art: false,
            },
            current_screen: CurrentScreen::SuggestingDirs,
            currently_editing: None,
            menu_items: vec![
                MenuItem::Model,
                MenuItem::InputPath,
                MenuItem::Threshold,
                MenuItem::BatchSize,
                MenuItem::ShowAsciiArt,
                MenuItem::Start,
            ],
            menu_index: 0,
            input_text: String::new(),
            progress: 0.0,
            status_message: String::from("Ready to start."),
            rx: None,
            is_error: false,
            suggested_dirs,
            selected_dirs: Vec::new(),
            suggestion_index: 0,
            show_ascii_art: false,
            current_frame: None,
            logs: Vec::new(),
            processed_image_paths: Vec::new(),
            current_image_index: 0,
        }
    }
}

impl App {
    /// Runs the main application loop.
    pub async fn run<B: Backend>(&mut self, terminal: &mut Terminal<B>) -> io::Result<()> {
        while self.current_screen != CurrentScreen::Exiting {
            terminal.draw(|f| ui::draw(f, self))?;
            self.handle_events()?;
            self.handle_progress_updates();
        }
        Ok(())
    }

    /// Handles progress updates from the processing thread.
    fn handle_progress_updates(&mut self) {
        if let Some(rx) = self.rx.as_mut() {
            if let Ok(update) = rx.try_recv() {
                match update {
                    ProgressUpdate::Message(msg) => {
                        self.status_message = msg.clone();
                        self.logs.push(msg);
                        if self.logs.len() > 100 {
                            self.logs.remove(0);
                        }
                    }
                    ProgressUpdate::Progress(p) => self.progress = p,
                    ProgressUpdate::Error(e) => {
                        self.status_message = format!("Error: {}", e);
                        self.logs.push(self.status_message.clone());
                        self.is_error = true;
                        self.current_screen = CurrentScreen::Finished;
                        self.rx = None;
                    }
                    ProgressUpdate::ImageProcessed(path) => {
                        let is_at_end = self.processed_image_paths.is_empty()
                            || self.current_image_index == self.processed_image_paths.len() - 1;
                        self.processed_image_paths.push(path);
                        if is_at_end {
                            self.current_image_index = self.processed_image_paths.len() - 1;
                            self.update_current_frame_from_path();
                        }
                    }
                    ProgressUpdate::Complete => {
                        self.status_message = "Processing complete!".to_string();
                        self.logs.push(self.status_message.clone());
                        self.is_error = false;
                        self.progress = 1.0;
                        self.current_screen = CurrentScreen::Finished;
                        self.rx = None;
                    }
                }
            }
        }
    }

    /// Handles user input events.
    fn handle_events(&mut self) -> io::Result<()> {
        if event::poll(std::time::Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    // Global key handlers
                    match key.code {
                        KeyCode::Char('a') | KeyCode::Left => {
                            self.scroll_left();
                            return Ok(());
                        }
                        KeyCode::Char('d') | KeyCode::Right => {
                            self.scroll_right();
                            return Ok(());
                        }
                        _ => {}
                    }

                    // Screen-specific key handlers
                    match self.current_screen {
                        CurrentScreen::SuggestingDirs => {
                            self.handle_suggesting_dirs_events(key.code)
                        }
                        CurrentScreen::Main => self.handle_main_screen_events(key.code),
                        CurrentScreen::Editing => self.handle_editing_screen_events(key.code),
                        CurrentScreen::Processing if key.code == KeyCode::Char('q') => {
                            self.current_screen = CurrentScreen::Main;
                            self.rx = None; // This will drop the sender, stopping the process
                        }
                        CurrentScreen::Finished if key.code == KeyCode::Enter => {
                            self.current_screen = CurrentScreen::Main;
                            self.status_message = "Ready to start.".to_string();
                            self.progress = 0.0;
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(())
    }

    /// Handles events for the directory suggestion screen.
    fn handle_suggesting_dirs_events(&mut self, key_code: KeyCode) {
        match key_code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.suggestion_index = self.suggestion_index.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if !self.suggested_dirs.is_empty() {
                    self.suggestion_index =
                        (self.suggestion_index + 1).min(self.suggested_dirs.len() - 1);
                }
            }
            KeyCode::Char(' ') => {
                if let Some(dir) = self.suggested_dirs.get(self.suggestion_index) {
                    if let Some(pos) = self.selected_dirs.iter().position(|x| x == dir) {
                        self.selected_dirs.remove(pos);
                    } else {
                        self.selected_dirs.push(dir.clone());
                    }
                }
            }
            KeyCode::Enter if !self.selected_dirs.is_empty() => {
                if let Some(dir_str) = self.selected_dirs[0].to_str() {
                    self.config.input_path = dir_str.to_string();
                }
                self.current_screen = CurrentScreen::Main;
            }
            KeyCode::Char('q') => {
                self.current_screen = CurrentScreen::Exiting;
            }
            _ => {}
        }
    }

    /// Handles events for the main menu screen.
    fn handle_main_screen_events(&mut self, key_code: KeyCode) {
        match key_code {
            KeyCode::Char('q') => self.current_screen = CurrentScreen::Exiting,
            KeyCode::Up | KeyCode::Char('k') => {
                self.menu_index = self.menu_index.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.menu_index = (self.menu_index + 1).min(self.menu_items.len() - 1);
            }
            KeyCode::Enter => self.handle_menu_selection(),
            _ => {}
        }
    }

    /// Handles the selection of a menu item.
    fn handle_menu_selection(&mut self) {
        let current_item = self.menu_items[self.menu_index];
        match current_item {
            MenuItem::Start => self.start_processing(),
            MenuItem::Model => self.config.model = self.config.model.next(),
            MenuItem::ShowAsciiArt => {
                self.show_ascii_art = !self.show_ascii_art;
                self.config.show_ascii_art = self.show_ascii_art;
            }
            _ => self.start_editing(current_item),
        }
    }

    /// Starts the background processing thread.
    fn start_processing(&mut self) {
        self.current_screen = CurrentScreen::Processing;
        self.progress = 0.0;
        self.status_message = "Starting...".to_string();

        let (tx, rx) = mpsc::channel(100);
        self.rx = Some(rx);

        let config = self.config.clone();
        let selected_dirs = self.selected_dirs.clone();

        tokio::spawn(async move {
            if let Err(e) = run_full_process(config, selected_dirs, tx.clone()).await {
                let _ = tx.send(ProgressUpdate::Error(e.to_string())).await;
            }
        });
    }

    /// Enters editing mode for a specific menu item.
    fn start_editing(&mut self, item: MenuItem) {
        self.currently_editing = Some(item);
        self.input_text = match item {
            MenuItem::InputPath => self.config.input_path.clone(),
            MenuItem::Threshold => self.config.threshold.to_string(),
            MenuItem::BatchSize => self.config.batch_size.to_string(),
            _ => String::new(),
        };
        self.current_screen = CurrentScreen::Editing;
    }

    /// Handles events for the editing screen.
    fn handle_editing_screen_events(&mut self, key_code: KeyCode) {
        match key_code {
            KeyCode::Enter => self.finish_editing(),
            KeyCode::Char(c) => self.input_text.push(c),
            KeyCode::Backspace => {
                self.input_text.pop();
            }
            KeyCode::Esc => {
                self.current_screen = CurrentScreen::Main;
                self.currently_editing = None;
            }
            _ => {}
        }
    }

    /// Finishes editing and updates the configuration.
    fn finish_editing(&mut self) {
        if let Some(editing) = self.currently_editing {
            match editing {
                MenuItem::InputPath => self.config.input_path = self.input_text.clone(),
                MenuItem::Threshold => {
                    self.config.threshold = self.input_text.parse().unwrap_or(self.config.threshold);
                }
                MenuItem::BatchSize => {
                    self.config.batch_size = self.input_text.parse().unwrap_or(self.config.batch_size);
                }
                _ => {}
            }
        }
        self.current_screen = CurrentScreen::Main;
        self.currently_editing = None;
        self.input_text.clear();
    }

    fn scroll_left(&mut self) {
        if !self.processed_image_paths.is_empty() {
            self.current_image_index = self.current_image_index.saturating_sub(1);
            self.update_current_frame_from_path();
        }
    }

    fn scroll_right(&mut self) {
        if !self.processed_image_paths.is_empty() {
            self.current_image_index =
                (self.current_image_index + 1).min(self.processed_image_paths.len() - 1);
            self.update_current_frame_from_path();
        }
    }

    fn update_current_frame_from_path(&mut self) {
        if let Some(path) = self.processed_image_paths.get(self.current_image_index) {
            if let Ok(img) = image::open(path) {
                self.current_frame = Some(img);
            }
        }
    }

    // Accessors
    pub fn menu_index(&self) -> usize {
        self.menu_index
    }
    pub fn menu_items(&self) -> &[MenuItem] {
        &self.menu_items
    }
    pub fn config(&self) -> &AppConfig {
        &self.config
    }
    pub fn is_editing(&self) -> bool {
        self.current_screen == CurrentScreen::Editing
    }
    pub fn input_text(&self) -> &str {
        &self.input_text
    }
    pub fn currently_editing(&self) -> Option<MenuItem> {
        self.currently_editing
    }
    pub fn current_screen(&self) -> &CurrentScreen {
        &self.current_screen
    }
}
