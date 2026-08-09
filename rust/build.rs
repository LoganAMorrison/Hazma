//! Link arguments for the extension module.
//!
//! On macOS a Python extension must be linked with
//! `-undefined dynamic_lookup`: CPython's symbols are absent at link
//! time and resolved by the interpreter that `dlopen`s the module.
//! Without it a plain `cargo build` fails with a wall of undefined
//! `_Py*` symbols even though the same crate builds fine through
//! setuptools-rust, which passes the flags itself.
//!
//! `rustc-cdylib-link-arg` applies only when a cdylib is being linked,
//! so the `cargo test` harness — which links libpython for real, under
//! `--no-default-features` — is unaffected.

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo::rustc-cdylib-link-arg=-undefined");
        println!("cargo::rustc-cdylib-link-arg=dynamic_lookup");
    }
}
