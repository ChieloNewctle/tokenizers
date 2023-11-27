use serde::{Deserialize, Serialize};

use crate::tokenizer::Trainer;

use super::GreedyTokenizer;

#[derive(Clone, Serialize, Deserialize)]
pub struct GTTrainer;

// TODO: Use a general suffix automaton to generate the vocabulary.
// Consider using suffix parents, solid edges, borders, cycles, etc.
//
// ababab...ababab: 1 * ab, 2 * ab, 4 * ab, 8 * ab, ...
//
// a state in the suffix automaton of the given sequences
// represents a set of ending position of some substring,
// which is the longest substring ending at the position.

impl Trainer for GTTrainer {
    type Model = GreedyTokenizer;

    fn should_show_progress(&self) -> bool {
        false
    }

    fn train(&self, _model: &mut Self::Model) -> crate::Result<Vec<crate::AddedToken>> {
        Err("Currently GreedyTokenizerTrainer is not implemented".into())
    }

    fn feed<I, S, F>(&mut self, _iterator: I, _process: F) -> crate::Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> crate::Result<Vec<String>> + Sync,
    {
        Err("Currently GreedyTokenizerTrainer is not implemented".into())
    }
}
