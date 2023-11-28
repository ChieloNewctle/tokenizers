pub mod model;
pub mod serialization;
pub mod trainer;

pub use model::{GreedyTokenizer, GreedyTokenizerBuilder};
pub use trainer::GTTrainer;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Unk token ID `{0}` is out of the vocabulary (size: `{1}`")]
    UnkTokenIDOutOfVocabulary(u32, usize),
    #[error("Unk token ID is not set, needed when tokenizing `{0:?}`")]
    UnkTokenIDNotSet(Box<[u8]>),
}
