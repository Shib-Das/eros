use clap::{Parser, Subcommand};
use std::fmt::{self, Display};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Process media files in a specified directory
    Process {
        /// The path to the directory containing media files to process
        #[arg(short, long)]
        path: String,

        /// The confidence threshold for tagging
        #[arg(short, long, default_value_t = 0.35)]
        threshold: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum V3Model {
    VitLarge,
    Eva02Large,
    #[default]
    SwinV2,
}


impl V3Model {
    pub fn repo_id(&self) -> String {
        match self {
            V3Model::VitLarge => "SmilingWolf/wd-vit-large-tagger-v3".to_string(),
            V3Model::Eva02Large => "SmilingWolf/wd-eva02-large-tagger-v3".to_string(),
            V3Model::SwinV2 => "SmilingWolf/wd-swinv2-tagger-v3".to_string(),
        }
    }

    pub fn next(&self) -> Self {
        match self {
            V3Model::VitLarge => V3Model::Eva02Large,
            V3Model::Eva02Large => V3Model::SwinV2,
            V3Model::SwinV2 => V3Model::VitLarge,
        }
    }
}

impl Display for V3Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let model_name = match self {
            V3Model::VitLarge => "ViT-Large",
            V3Model::Eva02Large => "Eva02-Large",
            V3Model::SwinV2 => "SwinV2",
        };
        write!(f, "{}", model_name)
    }
}