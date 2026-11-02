fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto/engine.proto");

    tonic_prost_build::compile_protos("proto/engine.proto")?;

    Ok(())
}
