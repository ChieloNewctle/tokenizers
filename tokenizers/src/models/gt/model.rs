use std::{
    collections::HashMap,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
};

use general_sam::{
    tokenize::OwnedGeneralSAM, trie::Trie, BTreeTransTable, BoxBisectTable, GeneralSAM,
    GreedyTokenizer as SAMGreedyTokenizer,
};

use crate::tokenizer::{Model, Result, Token};

use super::GTTrainer;

pub type Vocab = Vec<String>;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Config {
    pub(super) vocab: Vocab,
    pub(super) unk_token_id: Option<u32>,
    pub(super) byte_fallback: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab: Default::default(),
            unk_token_id: Default::default(),
            byte_fallback: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GreedyTokenizerBuilder {
    pub(super) config: Config,
}

fn capture_byte_repr(token: &str) -> Option<u8> {
    if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
        u8::from_str_radix(&token[3..5], 16).ok()
    } else {
        None
    }
}

impl GreedyTokenizerBuilder {
    #[must_use]
    pub fn vocab(mut self, vocab: Vocab) -> Self {
        self.config.vocab = vocab;
        self
    }

    #[must_use]
    pub fn unk_token_id(mut self, unk_token_id: u32) -> Self {
        self.config.unk_token_id = Some(unk_token_id);
        self
    }

    #[must_use]
    pub fn byte_fallback(mut self, flag: bool) -> Self {
        self.config.byte_fallback = flag;
        self
    }

    pub fn build(self) -> Result<GreedyTokenizer> {
        if let Some(unk_token_id) = self.config.unk_token_id {
            if unk_token_id as usize >= self.config.vocab.len() {
                return Err(super::Error::UnkTokenIDOutOfVocabulary(
                    unk_token_id,
                    self.config.vocab.len(),
                )
                .into());
            }
        }

        let mut trie: Trie<BTreeTransTable<_>> = Trie::default();
        let mut token_id_in_trie_map = HashMap::<usize, u32>::new();
        for (i, token) in self.config.vocab.iter().enumerate() {
            let k = if let Some(byte_repr) = self
                .config
                .byte_fallback
                .then_some(())
                .and_then(|_| capture_byte_repr(token))
            {
                trie.insert_ref_iter([byte_repr].iter())
            } else {
                trie.insert_ref_iter(token.as_bytes().iter())
            };

            token_id_in_trie_map.insert(k, i as u32);
        }
        let trie: Trie<BoxBisectTable<_>> = trie.alter_trans_table();

        let mut token_id_in_trie = Vec::<u32>::new();
        token_id_in_trie.resize(trie.num_of_nodes(), self.config.vocab.len() as u32);
        token_id_in_trie_map.iter().for_each(|(k, i)| {
            token_id_in_trie[*k] = *i;
        });

        let sam = GeneralSAM::<BTreeTransTable<_>>::from_trie(trie.get_root_state());

        Ok(GreedyTokenizer {
            inner: WrappedSAMGreedyTokenizer::from_sam_and_trie(
                sam.alter_trans_table_into(),
                &trie,
                &token_id_in_trie,
            ),
            token_to_id_map: self
                .config
                .vocab
                .iter()
                .enumerate()
                .map(|(u, v)| (v.to_owned(), u as u32))
                .collect(),
            config: self.config,
        })
    }
}

#[derive(Clone)]
struct WrappedSAMGreedyTokenizer(
    pub Arc<SAMGreedyTokenizer<BoxBisectTable<u8>, u32, OwnedGeneralSAM<BoxBisectTable<u8>>>>,
);

impl std::fmt::Debug for WrappedSAMGreedyTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WrappedSAMGreedyTokenizer")
            .field("tokenizer.sam", &self.0.get_sam_ref())
            .field("tokenizer.suffix_data", &self.0.get_suffix_data())
            .finish()
    }
}

impl WrappedSAMGreedyTokenizer {
    fn from_sam_and_trie(
        sam: GeneralSAM<BoxBisectTable<u8>>,
        trie: &Trie<BoxBisectTable<u8>>,
        token_id_in_trie: &[u32],
    ) -> Self {
        Self(Arc::new(SAMGreedyTokenizer::build_from_sam(
            sam,
            trie.get_root_state(),
            |trie_node| token_id_in_trie[trie_node.node_id],
        )))
    }
}

#[derive(Clone, Debug)]
pub struct GreedyTokenizer {
    pub(super) config: Config,
    token_to_id_map: HashMap<String, u32>,
    inner: WrappedSAMGreedyTokenizer,
}

impl PartialEq for GreedyTokenizer {
    fn eq(&self, other: &Self) -> bool {
        self.config == other.config
    }
}

impl Model for GreedyTokenizer {
    type Trainer = GTTrainer;

    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        let tokens = self.inner.0.tokenize(
            sequence.bytes(),
            &self
                .config
                .unk_token_id
                .unwrap_or_else(|| self.get_vocab_size() as u32),
        );
        let mut res = Vec::new();
        let mut cur_pos = 0;
        for (token_id, chunk_size) in tokens {
            res.push(Token {
                id: token_id,
                value: self.id_to_token(token_id).ok_or_else(|| {
                    super::Error::UnkTokenIDNotSet(
                        sequence[cur_pos..cur_pos + chunk_size].bytes().collect(),
                    )
                })?,
                offsets: (cur_pos, cur_pos + chunk_size),
            });
            cur_pos += chunk_size;
        }
        Ok(res)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id_map.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.config.vocab.get(id as usize).map(|x| x.to_owned())
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        let mut map = HashMap::new();
        self.config.vocab.iter().enumerate().for_each(|(i, token)| {
            map.insert(token.to_owned(), i as u32);
        });
        map
    }

    fn get_vocab_size(&self) -> usize {
        self.config.vocab.len()
    }

    fn save(
        &self,
        folder: &std::path::Path,
        prefix: Option<&str>,
    ) -> Result<Vec<std::path::PathBuf>> {
        let vocab_file_name = match prefix {
            Some(prefix) => format!("{}-vocab.json", prefix),
            None => "vocab.json".to_string(),
        };

        let vocab_path: PathBuf = [folder, Path::new(vocab_file_name.as_str())]
            .iter()
            .collect();

        let mut vocab_file = File::create(&vocab_path)?;
        let serialized = serde_json::to_string(&self.config.vocab)?;
        vocab_file.write_all(serialized.as_bytes())?;

        Ok(vec![vocab_path])
    }

    fn get_trainer(&self) -> Self::Trainer {
        Self::Trainer {}
    }
}
