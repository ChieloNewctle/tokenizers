use super::model::{GreedyTokenizer, GreedyTokenizerBuilder};
use serde::{
    de::{MapAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize, Serializer,
};

impl Serialize for GreedyTokenizer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut model = serializer.serialize_struct("GreedyTokenizer", 4)?;

        model.serialize_field("type", "GreedyTokenizer")?;
        model.serialize_field("byte_fallback", &self.config.byte_fallback)?;
        model.serialize_field("unk_token_id", &self.config.unk_token_id)?;
        model.serialize_field("vocab", &self.config.vocab)?;

        model.end()
    }
}

impl<'de> Deserialize<'de> for GreedyTokenizer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "GreedyTokenizer",
            &["type", "unk_token_id", "byte_fallback", "vocab"],
            GreedyTokenizerVisitor,
        )
    }
}

struct GreedyTokenizerVisitor;
impl<'de> Visitor<'de> for GreedyTokenizerVisitor {
    type Value = GreedyTokenizer;

    fn expecting(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "struct GreedyTokenizer")
    }

    fn visit_map<V>(self, mut map: V) -> std::result::Result<Self::Value, V::Error>
    where
        V: MapAccess<'de>,
    {
        let mut builder = GreedyTokenizerBuilder::default();
        while let Some(key) = map.next_key::<String>()? {
            match key.as_ref() {
                "unk_token_id" => {
                    if let Some(unk) = map.next_value()? {
                        builder = builder.unk_token_id(unk);
                    }
                }
                "byte_fallback" => {
                    if let Some(flag) = map.next_value()? {
                        builder = builder.byte_fallback(flag);
                    }
                }
                "vocab" => {
                    if let Some(vocab) = map.next_value()? {
                        builder = builder.vocab(vocab);
                    }
                }
                _ => {}
            }
        }
        builder.build().map_err(serde::de::Error::custom)
    }
}
