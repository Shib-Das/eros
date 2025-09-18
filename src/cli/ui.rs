use crate::app::{App, MenuItem};
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Clear, Gauge, List, ListItem, Paragraph},
};

pub fn draw(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .constraints([
            Constraint::Length(3), // Title
            Constraint::Min(0),    // Main content
            Constraint::Length(3), // Footer
        ])
        .split(f.size());

    let title = Paragraph::new("Eros Image Tagger")
        .style(Style::default().fg(Color::LightCyan))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(title, chunks[0]);

    render_menu(f, app, chunks[1]);

    let footer_text = "Use ↑/↓ or j/k to navigate, ↩ to select/edit, 'q' to quit.";
    let footer = Paragraph::new(footer_text)
        .style(Style::default().fg(Color::Gray))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(footer, chunks[2]);

    if app.is_editing() {
        render_popup(f, app);
    }
}

fn render_menu(f: &mut Frame, app: &App, area: Rect) {
    let config = app.config();
    let menu_items_widget: Vec<ListItem> = app
        .menu_items()
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let text = match item {
                MenuItem::Model => format!("Model: < {} >", config.model.to_string()),
                MenuItem::InputPath => format!("Input Path: {}", config.input_path),
                MenuItem::Threshold => format!("Threshold: {}", config.threshold),
                MenuItem::BatchSize => format!("Batch Size: {}", config.batch_size),
                MenuItem::Start => "Start Processing".to_string(),
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

fn render_popup(f: &mut Frame, app: &App) {
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
    let area = centered_rect(60, 20, f.size());
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

pub fn draw_progress(f: &mut Frame, total: u64, current: u64) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(2)
        .constraints([Constraint::Length(3), Constraint::Min(0)].as_ref())
        .split(f.size());

    let title = Paragraph::new("Eros Image Tagger - Processing")
        .style(Style::default().fg(Color::LightCyan))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL));

    f.render_widget(title, chunks[0]);

    let progress_percent = if total > 0 {
        (current as f64 / total as f64 * 100.0) as u16
    } else {
        0
    };

    let gauge = Gauge::default()
        .block(Block::default().title("Progress").borders(Borders::ALL))
        .gauge_style(
            Style::default()
                .fg(Color::Green)
                .bg(Color::Black)
                .add_modifier(Modifier::ITALIC),
        )
        .percent(progress_percent)
        .label(format!("{}/{}", current, total));

    f.render_widget(gauge, chunks[1]);
}