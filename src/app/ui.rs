use crate::{
    app::{App, CurrentScreen, MenuItem},
    ascii,
};
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Clear, Gauge, List, ListItem, Paragraph},
};

pub fn draw(f: &mut Frame, app: &App) {
    let base_chunks = Layout::default()
        .constraints([
            Constraint::Length(3), // Title
            Constraint::Min(0),    // Main content
            Constraint::Length(5), // Log view
            Constraint::Length(3), // Footer
        ])
        .split(f.area());

    let title = Paragraph::new("Eros Image Tagger")
        .style(Style::default().fg(Color::LightCyan))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(title, base_chunks[0]);

    // Dispatch rendering for the main content area
    match app.current_screen() {
        CurrentScreen::SuggestingDirs => {
            render_suggesting_dirs_screen(f, app, base_chunks[1]);
        }
        CurrentScreen::Processing => {
            render_processing_screen(f, app, base_chunks[1]);
        }
        CurrentScreen::Main | CurrentScreen::Editing => {
            render_main_screen(f, app, base_chunks[1]);
        }
        CurrentScreen::Finished => {
            // Render main screen in the background and finished popup on top
            render_main_screen(f, app, base_chunks[1]);
            render_finished_popup(f, app);
        }
        _ => {}
    }

    render_log(f, app, base_chunks[2]);

    let footer_text =
        "Use ↑/↓ or j/k to navigate, ↩ to select, 'q' to quit. Use 'a'/← and 'd'/→ to scroll images.";
    let footer = Paragraph::new(footer_text)
        .style(Style::default().fg(Color::Gray))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(footer, base_chunks[3]);

    if app.is_editing() {
        render_editing_popup(f, app);
    }
}

fn render_suggesting_dirs_screen(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .constraints([
            Constraint::Min(0), // List
        ])
        .split(area);

    let items: Vec<ListItem> = app
        .suggested_dirs
        .iter()
        .enumerate()
        .map(|(i, dir)| {
            let is_selected = app.selected_dirs.contains(dir);
            let prefix = if is_selected { "[x] " } else { "[ ] " };
            let text = format!("{}{}", prefix, dir.to_string_lossy());
            let style = if i == app.suggestion_index {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            ListItem::new(text).style(style)
        })
        .collect();

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Suggested Directories"),
        )
        .highlight_symbol(">> ");

    f.render_widget(list, chunks[0]);
}

fn render_main_screen(f: &mut Frame, app: &App, area: Rect) {
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    render_menu(f, app, main_chunks[0]);
    render_ascii_art(f, app, main_chunks[1]);
}

fn render_menu(f: &mut Frame, app: &App, area: Rect) {
    let config = app.config();
    let menu_items_widget: Vec<ListItem> = app
        .menu_items()
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let text = match item {
                MenuItem::Model => format!("Model: < {} >", config.model),
                MenuItem::InputPath => format!("Input Path: {}", config.input_path),
                MenuItem::Threshold => format!("Threshold: {}", config.threshold),
                MenuItem::BatchSize => format!("Batch Size: {}", config.batch_size),
                MenuItem::ShowAsciiArt => {
                    format!("Show ASCII Art: < {} >", if app.show_ascii_art { "On" } else { "Off" })
                }
                MenuItem::Start => "Start Processing".to_string(),
                MenuItem::VideoPath => format!("Video Path: {}", config.video_path),
            };
            let style = if i == app.menu_index() {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            ListItem::new(text).style(style)
        })
        .collect();

    let list = List::new(menu_items_widget)
        .block(Block::default().borders(Borders::ALL).title("Menu"))
        .highlight_symbol(">> ");

    f.render_widget(list, area);
}

fn render_ascii_art(f: &mut Frame, app: &App, area: Rect) {
    if app.show_ascii_art {
        let art = if let Some(frame) = &app.current_frame {
            // Subtract border size from the area
            let inner_area = area.inner(Margin {
                vertical: 1,
                horizontal: 1,
            });
            ascii::create_ascii_art(frame, inner_area)
        } else {
            "Waiting for image...".to_string()
        };

        let title = if app.processed_image_paths.is_empty() {
            "ASCII Art".to_string()
        } else {
            format!(
                "ASCII Art - Image {}/{}",
                app.current_image_index + 1,
                app.processed_image_paths.len()
            )
        };

        let ascii_art_widget = Paragraph::new(art)
            .block(Block::default().borders(Borders::ALL).title(title))
            .alignment(Alignment::Center);
        f.render_widget(ascii_art_widget, area);
    } else {
        f.render_widget(
            Block::default()
                .borders(Borders::ALL)
                .title("Preview (Enable ASCII Art in Menu)"),
            area,
        );
    }
}

fn render_editing_popup(f: &mut Frame, app: &App) {
    let popup_title = match app.currently_editing() {
        Some(MenuItem::InputPath) => "Edit Input Path",
        Some(MenuItem::Threshold) => "Edit Threshold",
        Some(MenuItem::BatchSize) => "Edit Batch Size",
        _ => "Editing",
    };

    let block = Block::default()
        .title(popup_title)
        .borders(Borders::ALL)
        .style(Style::default().bg(Color::DarkGray));
    let area = centered_rect(60, 20, f.area());
    f.render_widget(Clear, area); //this clears the background
    f.render_widget(block, area);

    let popup_chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(2)
        .constraints([Constraint::Min(1), Constraint::Length(1)].as_ref())
        .split(area);

    let text_input = Paragraph::new(app.input_text())
        .style(Style::default().fg(Color::White))
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(text_input, popup_chunks[0]);

    let help_text = Paragraph::new("Press <Enter> to save, <Esc> to cancel")
        .style(Style::default().fg(Color::LightYellow));
    f.render_widget(help_text, popup_chunks[1]);
}

fn render_processing_screen(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Left side: Progress bar and status
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(45), // Spacer
            Constraint::Length(3),      // Gauge
            Constraint::Percentage(45), // Spacer
        ])
        .split(chunks[0]);

    let centered_area = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(10), // 10% margin on left
            Constraint::Percentage(80), // 80% width for gauge
            Constraint::Percentage(10), // 10% margin on right
        ])
        .split(left_chunks[1])[1];

    let label = format!("{} ({:.1}%)", &app.status_message, app.progress * 100.0);

    let gauge = Gauge::default()
        .block(Block::default().title("Progress").borders(Borders::ALL))
        .gauge_style(
            Style::default()
                .fg(Color::Green)
                .bg(Color::Black)
                .add_modifier(Modifier::ITALIC),
        )
        .percent((app.progress * 100.0) as u16)
        .label(label);

    f.render_widget(gauge, centered_area);

    // Right side: ASCII Art
    render_ascii_art(f, app, chunks[1]);
}

fn render_finished_popup(f: &mut Frame, app: &App) {
    let (popup_title, title_color) = if app.is_error {
        ("Error", Color::Red)
    } else {
        ("Success", Color::Green)
    };

    let block = Block::default()
        .title(popup_title)
        .borders(Borders::ALL)
        .style(
            Style::default()
                .bg(Color::DarkGray)
                .fg(title_color),
        );
    let area = centered_rect(80, 40, f.area());
    f.render_widget(Clear, area); //this clears the background
    f.render_widget(block, area);

    let popup_chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(2)
        .constraints([Constraint::Min(1), Constraint::Length(1)].as_ref())
        .split(area);

    let message = Paragraph::new(app.status_message.as_str())
        .wrap(ratatui::widgets::Wrap { trim: false })
        .style(Style::default().fg(Color::White));
    f.render_widget(message, popup_chunks[0]);

    let help_text = Paragraph::new("Press <Enter> to continue")
        .style(Style::default().fg(Color::LightYellow));
    f.render_widget(help_text, popup_chunks[1]);
}

fn render_log(f: &mut Frame, app: &App, area: Rect) {
    let log_messages: Vec<ListItem> = app
        .logs
        .iter()
        .map(|msg| ListItem::new(msg.as_str()))
        .collect();

    let logs_list = List::new(log_messages)
        .block(Block::default().borders(Borders::ALL).title("Logs"));

    f.render_widget(logs_list, area);
}

/// Helper function to create a centered rect using up certain percentage of the available rect `r`
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}