pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const GIT_SHA: &str = env!("PEGAFLOW_BUILD_GIT_SHA");
pub const BUILD_TIMESTAMP: &str = env!("PEGAFLOW_BUILD_TIMESTAMP");
pub const LONG_VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    "\n",
    "git: ",
    env!("PEGAFLOW_BUILD_GIT_SHA"),
    "\n",
    "built: ",
    env!("PEGAFLOW_BUILD_TIMESTAMP")
);
