use crate::args::V3Model;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::{backend::Backend, Terminal};
use std::io;

use super::ui;

#[derive(Debug, Default, Clone)]
pub struct AppConfig {
    pub model: V3Model,
    pub input_path: String,
    pub threshold: f32,
    pub batch_size: usize,
}

#[derive(Debug, PartialEq, Eq)]
enum CurrentScreen {
    Main,
    Editing,
    Exiting,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MenuItem {
    Model,
    InputPath,
    Threshold,
    BatchSize,
    Start,
}

pub struct App {
    config: AppConfig,
    current_screen: CurrentScreen,
    currently_editing: Option<MenuItem>,
    menu_items: Vec<MenuItem>,
    menu_index: usize,
    input_text: String,
}

impl Default for App {
    fn default() -> Self {
        Self {
            config: AppConfig {
                model: V3Model::SwinV2,
                input_path: "./images".to_string(),
                threshold: 0.35,
                batch_size: 1,
            },
            current_screen: CurrentScreen::Main,
            currently_editing: None,
            menu_items: vec![
                MenuItem::Model,
                MenuItem::InputPath,
                MenuItem::Threshold,
                MenuItem::BatchSize,
                MenuItem::Start,
            ],
            menu_index: 0,
            input_text: String::new(),
        }
    }
}

impl App {
    pub fn run<B: Backend>(&mut self, terminal: &mut Terminal<B>) -> io::Result<Option<AppConfig>> {
        loop {
            terminal.draw(|f| ui::draw(f, self))?;
            self.handle_events()?;

            if let CurrentScreen::Exiting = self.current_screen {
                if self.menu_items[self.menu_index] == MenuItem::Start {
                    return Ok(Some(self.config.clone()));
                } else {
                    return Ok(None);
                }
            }
        }
    }

    fn handle_events(&mut self) -> io::Result<()> {
        if let Event::Key(key) = event::read()? {
            if key.kind == KeyEventKind::Press {
                match self.current_screen {
                    CurrentScreen::Main => self.handle_main_screen_events(key.code),
                    CurrentScreen::Editing => self.handle_editing_screen_events(key.code),
                    CurrentScreen::Exiting => {}
                }
            }
        }
        Ok(())
    }

    fn handle_main_screen_events(&mut self, key_code: KeyCode) {
        match key_code {
            KeyCode::Char('q') => self.current_screen = CurrentScreen::Exiting,
            KeyCode::Up | KeyCode::Char('k') => {
                if self.menu_index > 0 {
                    self.menu_index -= 1;
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.menu_index < self.menu_items.len() - 1 {
                    self.menu_index += 1;
                }
            }
            KeyCode::Enter => {
                let current_item = self.menu_items[self.menu_index];
                if current_item == MenuItem::Start {
                    self.current_screen = CurrentScreen::Exiting;
                } else if current_item == MenuItem::Model {
                    self.config.model = self.config.model.next();
                } else {
                    self.currently_editing = Some(current_item);
                    self.input_text = match current_item {
                        MenuItem::InputPath => self.config.input_path.clone(),
                        MenuItem::Threshold => self.config.threshold.to_string(),
                        MenuItem::BatchSize => self.config.batch_size.to_string(),
                        _ => String::new(),
                    };
                    self.current_screen = CurrentScreen::Editing;
                }
            }
            _ => {}
        }
    }

    fn handle_editing_screen_events(&mut self, key_code: KeyCode) {
        match key_code {
            KeyCode::Enter => {
                if let Some(editing) = self.currently_editing {
                    match editing {
                        MenuItem::InputPath => self.config.input_path = self.input_text.clone(),
                        MenuItem::Threshold => {
                            if let Ok(val) = self.input_text.parse() {
                                self.config.threshold = val;
                            }
                        }
                        MenuItem::BatchSize => {
                            if let Ok(val) = self.input_text.parse() {
                                self.config.batch_size = val;
                            }
                        }
                        _ => {}
                    }
                }
                self.current_screen = CurrentScreen::Main;
                self.currently_editing = None;
                self.input_text.clear();
            }
            KeyCode::Char(c) => {
                self.input_text.push(c);
            }
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

    // ... getters for UI rendering ...
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
}