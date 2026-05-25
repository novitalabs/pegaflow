//! pegaflow-transfer build script.
//!
//! Generates FFI bindings for libibverbs when the `v2-rdma` feature is enabled.
//! CUDA driver/runtime bindings come from `cudarc`.

use std::{
    env,
    path::{Path, PathBuf},
};

fn find_package(
    provider: &str,
    env_var: &str,
    default_paths: &[&str],
    check_file: &str,
) -> PathBuf {
    println!("cargo:rerun-if-env-changed={}", env_var);
    env::var_os(env_var)
        .map(PathBuf::from)
        .into_iter()
        .chain(default_paths.iter().map(PathBuf::from))
        .find(|dir| dir.join(check_file).is_file())
        .unwrap_or_else(|| {
            panic!(
                "{provider}: required header `{check_file}` not found. \
                 Looked at `${env_var}` ({env_status}) and default paths {default_paths:?}. \
                 Hint: install the provider headers or set `{env_var}` to their install root.",
                env_status = env::var_os(env_var)
                    .map(|v| format!("set to {:?}", v))
                    .unwrap_or_else(|| "unset".to_string()),
            )
        })
}

fn build_libibverbs(out_dir: &Path, manifest: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let home = find_package(
        "libibverbs",
        "LIBIBVERBS_HOME",
        &["/usr"],
        "include/infiniband/verbs.h",
    );

    let wrapper = manifest.join("build_wrappers/libibverbs.h");
    let bindings = bindgen::Builder::default()
        .header(wrapper.to_string_lossy())
        .clang_arg(format!("-I{}/include", home.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .prepend_enum_name(false)
        .allowlist_item(r"(ibv_|IBV_|ib_|IB_).*")
        .derive_debug(false)
        .derive_default(true)
        .wrap_static_fns(true)
        .wrap_static_fns_path(out_dir.join("wrap_static_fns.c"))
        .allowlist_item(r"pthread_.*")
        .opaque_type(r"pthread_.*")
        .no_default(r"pthread_.*")
        .generate()
        .map_err(|e| format!("libibverbs bindgen failed: {}", e))?;
    bindings.write_to_file(out_dir.join("libibverbs-bindings.rs"))?;

    cc::Build::new()
        .file(out_dir.join("wrap_static_fns.c"))
        .include(home.join("include"))
        .include(manifest.join("build_wrappers"))
        .compile("wrap_static_fns");

    println!("cargo:rustc-link-search=native={}/lib", home.display());
    println!("cargo:rustc-link-lib=ibverbs");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    println!("cargo:rerun-if-changed=build_wrappers/libibverbs.h");

    build_libibverbs(&out_dir, &manifest)?;
    Ok(())
}
