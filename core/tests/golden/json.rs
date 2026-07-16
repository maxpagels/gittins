//! A minimal JSON parser for reading `spec/golden.json` — kept in the test
//! tree so the crate itself stays zero-dependency. Numbers are kept as their
//! raw literals and parsed on demand: hashes as u64 (they exceed 2^53), and
//! floats via `str::parse::<f64>`, which is correctly rounded — the corpus
//! writes shortest round-trip reprs, so parsing recovers the exact bits.

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(String),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn parse(text: &str) -> Json {
        let mut p = Parser {
            bytes: text.as_bytes(),
            pos: 0,
        };
        let value = p.value();
        p.skip_ws();
        assert!(p.pos == p.bytes.len(), "trailing content after JSON value");
        value
    }

    pub fn get(&self, key: &str) -> &Json {
        match self {
            Json::Obj(entries) => entries
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v)
                .unwrap_or_else(|| panic!("missing key {key:?}")),
            _ => panic!("get({key:?}) on non-object"),
        }
    }

    pub fn entries(&self) -> &[(String, Json)] {
        match self {
            Json::Obj(entries) => entries,
            _ => panic!("not an object"),
        }
    }

    pub fn arr(&self) -> &[Json] {
        match self {
            Json::Arr(items) => items,
            _ => panic!("not an array"),
        }
    }

    pub fn str_(&self) -> &str {
        match self {
            Json::Str(s) => s,
            _ => panic!("not a string"),
        }
    }

    pub fn f64_(&self) -> f64 {
        match self {
            Json::Num(raw) => raw.parse().expect("bad float literal"),
            _ => panic!("not a number"),
        }
    }

    pub fn u64_(&self) -> u64 {
        match self {
            Json::Num(raw) => raw.parse().expect("bad u64 literal"),
            _ => panic!("not a number"),
        }
    }

    pub fn usize_(&self) -> usize {
        self.u64_() as usize
    }
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len()
            && matches!(self.bytes[self.pos], b' ' | b'\t' | b'\n' | b'\r')
        {
            self.pos += 1;
        }
    }

    fn peek(&self) -> u8 {
        self.bytes[self.pos]
    }

    fn eat(&mut self, expected: u8) {
        assert!(
            self.pos < self.bytes.len() && self.bytes[self.pos] == expected,
            "expected {:?} at byte {}",
            expected as char,
            self.pos
        );
        self.pos += 1;
    }

    fn literal(&mut self, word: &str, value: Json) -> Json {
        assert!(
            self.bytes[self.pos..].starts_with(word.as_bytes()),
            "bad literal at byte {}",
            self.pos
        );
        self.pos += word.len();
        value
    }

    fn value(&mut self) -> Json {
        self.skip_ws();
        match self.peek() {
            b'n' => self.literal("null", Json::Null),
            b't' => self.literal("true", Json::Bool(true)),
            b'f' => self.literal("false", Json::Bool(false)),
            b'"' => Json::Str(self.string()),
            b'[' => {
                self.eat(b'[');
                let mut items = Vec::new();
                self.skip_ws();
                if self.peek() == b']' {
                    self.eat(b']');
                    return Json::Arr(items);
                }
                loop {
                    items.push(self.value());
                    self.skip_ws();
                    if self.peek() == b',' {
                        self.eat(b',');
                    } else {
                        self.eat(b']');
                        return Json::Arr(items);
                    }
                }
            }
            b'{' => {
                self.eat(b'{');
                let mut entries = Vec::new();
                self.skip_ws();
                if self.peek() == b'}' {
                    self.eat(b'}');
                    return Json::Obj(entries);
                }
                loop {
                    self.skip_ws();
                    let key = self.string();
                    self.skip_ws();
                    self.eat(b':');
                    entries.push((key, self.value()));
                    self.skip_ws();
                    if self.peek() == b',' {
                        self.eat(b',');
                    } else {
                        self.eat(b'}');
                        return Json::Obj(entries);
                    }
                }
            }
            _ => {
                let start = self.pos;
                while self.pos < self.bytes.len()
                    && matches!(self.bytes[self.pos], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
                {
                    self.pos += 1;
                }
                assert!(self.pos > start, "unexpected byte at {}", start);
                Json::Num(String::from_utf8(self.bytes[start..self.pos].to_vec()).unwrap())
            }
        }
    }

    fn string(&mut self) -> String {
        self.eat(b'"');
        let mut out = String::new();
        loop {
            let b = self.bytes[self.pos];
            match b {
                b'"' => {
                    self.pos += 1;
                    return out;
                }
                b'\\' => {
                    self.pos += 1;
                    let esc = self.bytes[self.pos];
                    self.pos += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let unit = self.hex4();
                            let ch = if (0xD800..0xDC00).contains(&unit) {
                                // High surrogate: a \uXXXX low surrogate follows.
                                self.eat(b'\\');
                                self.eat(b'u');
                                let low = self.hex4();
                                let c = 0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00);
                                char::from_u32(c).expect("bad surrogate pair")
                            } else {
                                char::from_u32(unit).expect("bad unicode escape")
                            };
                            out.push(ch);
                        }
                        _ => panic!("bad escape at byte {}", self.pos - 1),
                    }
                }
                _ => {
                    // Copy one UTF-8 encoded character verbatim.
                    let len = match b {
                        0x00..=0x7F => 1,
                        0xC0..=0xDF => 2,
                        0xE0..=0xEF => 3,
                        _ => 4,
                    };
                    out.push_str(
                        std::str::from_utf8(&self.bytes[self.pos..self.pos + len]).unwrap(),
                    );
                    self.pos += len;
                }
            }
        }
    }

    fn hex4(&mut self) -> u32 {
        let s = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4]).unwrap();
        self.pos += 4;
        u32::from_str_radix(s, 16).expect("bad \\u escape")
    }
}
