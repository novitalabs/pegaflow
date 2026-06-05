use cudarc::runtime::{result as cuda_runtime, sys as cuda_runtime_sys};
use log::info;

pub(crate) fn preflight() -> Result<(), CudaVersionError> {
    let build_version = cuda_runtime_sys::CUDART_VERSION as i32;
    info!(
        "CUDA preflight: build={}, checking CUDA Runtime",
        format_cuda_version(build_version)
    );

    let result = catch_unwind_silent(|| {
        let runtime_version = cuda_runtime::version::get_runtime_version()?;
        let driver_version = cuda_runtime::version::get_driver_version()?;
        Ok::<_, cuda_runtime::RuntimeError>((runtime_version, driver_version))
    });

    let (runtime_version, driver_version) = match result {
        Ok(Ok(versions)) => versions,
        Ok(Err(err)) => {
            return Err(CudaVersionError(format!(
                "CUDA preflight failed: build={}, version query returned {err}",
                format_cuda_version(build_version),
            )));
        }
        Err(payload) => {
            log::debug!(
                "CUDA runtime symbol loading failed during preflight: {}",
                panic_payload_to_string(payload.as_ref()),
            );
            return Err(CudaVersionError(format!(
                "CUDA runtime mismatch: build={}, failed to load runtime symbols. \
                 Rebuild with the matching cuda feature or fix LD_LIBRARY_PATH/CUDA_HOME.",
                format_cuda_version(build_version),
            )));
        }
    };

    info!(
        "CUDA preflight: runtime={}, driver={}",
        format_cuda_version(runtime_version),
        format_cuda_version(driver_version)
    );

    if !is_compatible_cuda_runtime(build_version, runtime_version) {
        return Err(CudaVersionError(format!(
            "CUDA runtime mismatch: build={}, runtime={}. \
             Rebuild with the matching cuda feature or fix LD_LIBRARY_PATH/CUDA_HOME.",
            format_cuda_version(build_version),
            format_cuda_version(runtime_version),
        )));
    }

    Ok(())
}

pub(crate) struct CudaVersionError(String);

impl std::fmt::Display for CudaVersionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::fmt::Debug for CudaVersionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for CudaVersionError {}

fn catch_unwind_silent<F, T>(f: F) -> std::thread::Result<T>
where
    F: FnOnce() -> T + std::panic::UnwindSafe,
{
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(f);
    std::panic::set_hook(hook);
    result
}

fn panic_payload_to_string(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else if let Some(msg) = payload.downcast_ref::<&str>() {
        (*msg).to_string()
    } else {
        "<unknown panic payload>".to_string()
    }
}

fn is_compatible_cuda_runtime(build_version: i32, runtime_version: i32) -> bool {
    cuda_major(build_version) == cuda_major(runtime_version) && runtime_version >= build_version
}

fn cuda_major(version: i32) -> i32 {
    version / 1000
}

fn format_cuda_version(version: i32) -> String {
    let major = cuda_major(version);
    let minor = (version % 1000) / 10;
    format!("{major}.{minor} ({version})")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatible_runtime_must_match_major_and_not_be_older() {
        assert!(is_compatible_cuda_runtime(12080, 12080));
        assert!(is_compatible_cuda_runtime(12080, 12090));
        assert!(is_compatible_cuda_runtime(13000, 13010));

        assert!(!is_compatible_cuda_runtime(12080, 12070));
        assert!(!is_compatible_cuda_runtime(12080, 13000));
        assert!(!is_compatible_cuda_runtime(13000, 12080));
    }

    #[test]
    fn cuda_versions_are_formatted_for_logs_and_errors() {
        assert_eq!(format_cuda_version(12080), "12.8 (12080)");
        assert_eq!(format_cuda_version(13000), "13.0 (13000)");
        assert_eq!(format_cuda_version(13010), "13.1 (13010)");
    }
}
