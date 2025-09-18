#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V3Model {
    Vit,
    SwinV2,
    Convnext,
    VitLarge,
    Eva02Large,
}

impl Default for V3Model {
    fn default() -> Self {
        V3Model::SwinV2
    }
}

impl V3Model {
    pub fn repo_id(&self) -> String {
        match self {
            V3Model::Vit => "SmilingWolf/wd-vit-tagger-v3".to_string(),
            V3Model::SwinV2 => "SmilingWolf/wd-swinv2-tagger-v3".to_string(),
            V3Model::Convnext => "SmilingWolf/wd-convnext-tagger-v3".to_string(),
            V3Model::VitLarge => "SmilingWolf/wd-vit-large-tagger-v3".to_string(),
            V3Model::Eva02Large => "SmilingWolf/wd-eva02-large-tagger-v3".to_string(),
        }
    }

    pub fn next(&self) -> Self {
        match self {
            V3Model::Vit => V3Model::SwinV2,
            V3Model::SwinV2 => V3Model::Convnext,
            V3Model::Convnext => V3Model::VitLarge,
            V3Model::VitLarge => V3Model::Eva02Large,
            V3Model::Eva02Large => V3Model::Vit,
        }
    }
}

impl ToString for V3Model {
    fn to_string(&self) -> String {
        match self {
            V3Model::Vit => "ViT".to_string(),
            V3Model::SwinV2 => "SwinV2".to_string(),
            V3Model::Convnext => "ConvNeXT".to_string(),
            V3Model::VitLarge => "ViT-Large".to_string(),
            V3Model::Eva02Large => "Eva02-Large".to_string(),
        }
    }
}