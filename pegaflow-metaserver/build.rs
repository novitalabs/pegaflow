use std::process::Command;

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }

    let value = String::from_utf8(output.stdout).ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=PEGAFLOW_BUILD_GIT_SHA");
    println!("cargo:rerun-if-env-changed=PEGAFLOW_BUILD_TIMESTAMP");

    let git_sha = std::env::var("PEGAFLOW_BUILD_GIT_SHA")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| command_output("git", &["rev-parse", "--short=12", "HEAD"]))
        .unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=PEGAFLOW_BUILD_GIT_SHA={git_sha}");

    let build_timestamp = std::env::var("PEGAFLOW_BUILD_TIMESTAMP")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| command_output("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]))
        .unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=PEGAFLOW_BUILD_TIMESTAMP={build_timestamp}");
}
