// Binary wrapper for pegaflow-server
// This delegates to the pegaflow-server crate's run() function

fn main() -> Result<(), Box<dyn std::error::Error>> {
    pegaflow_server::run()
}
