//! The core's error surface. Every rejection the reference expresses as a
//! `ValueError` — invalid construction parameters, malformed candidates,
//! malformed serialized state — is a returned `Error` here, with the same
//! message text, so the two implementations reject identically. Nothing on
//! the public path panics: bindings sit directly on these functions, and a
//! panic must never cross an FFI boundary.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    message: String,
}

impl Error {
    /// Public so bindings can carry their own failures — a user callback
    /// raising, a malformed callback result — through the BYO callback
    /// types' `Err` side (decide.rs, api.rs).
    pub fn new(message: impl Into<String>) -> Error {
        Error {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}
